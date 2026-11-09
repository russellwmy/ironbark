//! The little of HTTP/1.1 the server needs: one request per connection, `Content-Length` bodies.

use std::io::{self, Read, Write};

/// Largest request head.
const MAX_HEAD: usize = 64 * 1024;
/// Largest request body: a long conversation with tool results.
const MAX_BODY: usize = 64 * 1024 * 1024;

/// A request.
#[derive(Debug)]
pub(crate) struct Request {
    pub(crate) method: String,
    /// Without the query.
    pub(crate) path: String,
    pub(crate) authorization: Option<String>,
    pub(crate) body: Vec<u8>,
}

/// Why a request could not be read.
#[derive(Debug)]
pub(crate) enum ReadError {
    /// The connection failed or closed: nobody to answer.
    Io,
    /// Answer with this status and message.
    Bad(u16, &'static str),
}

impl From<io::Error> for ReadError {
    fn from(_: io::Error) -> Self {
        Self::Io
    }
}

/// Read one request from `s` (a `100 Continue` is written to it if the client waits for one).
pub(crate) fn read_request<S: Read + Write>(s: &mut S) -> Result<Request, ReadError> {
    let mut data = Vec::new();
    let mut chunk = [0u8; 8192];
    let head_end = loop {
        if let Some(i) = data.windows(4).position(|w| w == b"\r\n\r\n") {
            break i;
        }
        if data.len() > MAX_HEAD {
            return Err(ReadError::Bad(431, "request head too large"));
        }
        let n = s.read(&mut chunk)?;
        if n == 0 {
            return Err(ReadError::Io);
        }
        data.extend_from_slice(&chunk[..n]);
    };
    let head = std::str::from_utf8(&data[..head_end]).map_err(|_| ReadError::Bad(400, "request head is not UTF-8"))?;
    let mut lines = head.split("\r\n");
    let mut first = lines.next().unwrap_or("").split(' ');
    let (Some(method), Some(target)) = (first.next(), first.next()) else {
        return Err(ReadError::Bad(400, "bad request line"));
    };
    let (mut length, mut authorization, mut expect) = (0usize, None, false);
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match name.trim().to_ascii_lowercase().as_str() {
            "content-length" => {
                length = value.parse().map_err(|_| ReadError::Bad(400, "bad Content-Length"))?;
            }
            "transfer-encoding" => return Err(ReadError::Bad(411, "send a Content-Length")),
            "authorization" => authorization = Some(value.to_owned()),
            "expect" => expect = value.eq_ignore_ascii_case("100-continue"),
            _ => {}
        }
    }
    if length > MAX_BODY {
        return Err(ReadError::Bad(413, "request body too large"));
    }
    let method = method.to_owned();
    let path = target.split('?').next().unwrap_or(target).to_owned();
    if expect {
        s.write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?;
    }
    let mut body = data.split_off(head_end + 4);
    while body.len() < length {
        let n = s.read(&mut chunk)?;
        if n == 0 {
            return Err(ReadError::Io);
        }
        body.extend_from_slice(&chunk[..n]);
    }
    body.truncate(length);
    Ok(Request {
        method,
        path,
        authorization,
        body,
    })
}

const fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        411 => "Length Required",
        413 => "Content Too Large",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Error",
    }
}

/// Write a whole JSON response.
pub(crate) fn write_json<S: Write>(s: &mut S, status: u16, body: &str) -> io::Result<()> {
    write!(
        s,
        "HTTP/1.1 {status} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        reason(status),
        body.len()
    )?;
    s.flush()
}

/// Start a server-sent event stream; it ends when the connection closes.
pub(crate) fn write_stream_head<S: Write>(s: &mut S) -> io::Result<()> {
    s.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
    )?;
    s.flush()
}

/// Write one server-sent event.
pub(crate) fn write_event<S: Write>(s: &mut S, data: &str) -> io::Result<()> {
    write!(s, "data: {data}\n\n")?;
    s.flush()
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    /// A stream reading `input`, its writes kept.
    struct Duplex(Cursor<Vec<u8>>, Vec<u8>);

    impl Read for Duplex {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.0.read(buf)
        }
    }

    impl Write for Duplex {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.1.write(buf)
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn read(input: &str) -> (Result<Request, ReadError>, Vec<u8>) {
        let mut d = Duplex(Cursor::new(input.as_bytes().to_vec()), Vec::new());
        let r = read_request(&mut d);
        (r, d.1)
    }

    #[test]
    fn reads_a_request_with_a_body() {
        let (r, written) =
            read("POST /v1/x?a=1 HTTP/1.1\r\ncontent-LENGTH: 5\r\nAuthorization: Bearer k\r\n\r\nhelloEXTRA");
        let r = r.unwrap();
        assert_eq!((r.method.as_str(), r.path.as_str()), ("POST", "/v1/x"));
        assert_eq!(r.authorization.as_deref(), Some("Bearer k"));
        assert_eq!(r.body, b"hello");
        assert_eq!(written, b"");
    }

    #[test]
    fn answers_expect_continue() {
        let (r, written) = read("POST / HTTP/1.1\r\nExpect: 100-continue\r\nContent-Length: 2\r\n\r\nhi");
        assert_eq!(r.unwrap().body, b"hi");
        assert!(written.starts_with(b"HTTP/1.1 100"));
    }

    #[test]
    fn refuses_what_it_cannot_read() {
        let status = |input: &str| match read(input).0 {
            Err(ReadError::Bad(s, _)) => s,
            other => unreachable!("{other:?}"),
        };
        assert_eq!(status("POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"), 411);
        assert_eq!(status("POST / HTTP/1.1\r\nContent-Length: 99999999999\r\n\r\n"), 413);
        assert_eq!(status("POST / HTTP/1.1\r\nContent-Length: x\r\n\r\n"), 400);
        assert_eq!(status(&format!("GET / HTTP/1.1\r\nX: {}\r\n", "a".repeat(MAX_HEAD + 9000))), 431);
        assert!(matches!(read("POST / HTTP/1.1\r\nContent-Length: 9\r\n\r\nshort").0, Err(ReadError::Io)));
        assert!(matches!(read("").0, Err(ReadError::Io)));
    }
}
