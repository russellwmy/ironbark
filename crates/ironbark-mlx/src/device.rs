//! What the hardware allows: physical RAM, the app memory limit (iOS), and the process
//! footprint, which is what iOS counts against that limit.

use std::ffi::c_void;
use std::sync::mpsc;
use std::time::Duration;

/// Memory facts of the running device.
#[derive(Debug, Clone, Copy)]
pub struct Hardware {
    /// Physical RAM in bytes.
    pub ram: u64,
    /// Most the process may use before the OS terminates it (iOS); `None` on macOS.
    pub app_limit: Option<u64>,
}

impl Hardware {
    /// Read the current device.
    #[must_use]
    pub fn detect() -> Self {
        Self {
            ram: ram(),
            app_limit: app_limit(),
        }
    }
}

/// Process footprint now and at its peak, in bytes.
#[derive(Debug, Clone, Copy)]
pub struct Footprint {
    /// Current footprint.
    pub now: u64,
    /// Highest footprint since the process started.
    pub peak: u64,
}

/// `task_vm_info` from `<mach/task_info.h>` up to revision 4 (`limit_bytes_remaining`).
#[repr(C, packed(4))]
#[derive(Default)]
struct VmInfo {
    virtual_size: u64,
    region_count: i32,
    page_size: i32,
    /// `resident_size` .. `compressed_lifetime`.
    sizes: [u64; 16],
    phys_footprint: u64,
    min_max_address: [u64; 2],
    ledger_phys_footprint_peak: i64,
    /// `ledger_purgeable_nonvolatile` .. `ledger_tag_neural_nofootprint_compressed`.
    ledgers: [i64; 20],
    limit_bytes_remaining: u64,
}

unsafe extern "C" {
    /// The task's own port (`<mach/mach_init.h>`), set at process start.
    static mach_task_self_: u32;
    /// `<mach/task.h>`; libc declares it for macOS only.
    fn task_info(task: u32, flavor: i32, info: *mut i32, count: *mut u32) -> i32;
}

/// The process footprint (`phys_footprint`), as iOS's memory limit counts it.
#[must_use]
pub fn footprint() -> Option<Footprint> {
    const TASK_VM_INFO: i32 = 22;
    let mut info = VmInfo::default();
    let mut count = (size_of::<VmInfo>() / 4) as u32;
    // SAFETY: a plain read of a port number the runtime writes once at start.
    let task = unsafe { mach_task_self_ };
    // SAFETY: `info` is a writable task_vm_info prefix of `count` 32-bit words.
    let rc = unsafe { task_info(task, TASK_VM_INFO, (&raw mut info).cast(), &raw mut count) };
    (rc == 0).then_some(Footprint {
        now: info.phys_footprint,
        peak: info.ledger_phys_footprint_peak.max(0) as u64,
    })
}

/// Highest footprint over a span: a thread samples [`footprint`] every millisecond until
/// [`PeakSampler::stop`]. The system's peak covers the whole process; this covers one run.
#[derive(Debug)]
pub struct PeakSampler {
    stop: mpsc::Sender<()>,
    thread: std::thread::JoinHandle<u64>,
}

impl PeakSampler {
    /// Start sampling.
    #[must_use]
    pub fn start() -> Self {
        let (stop, stopped) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            let mut peak = 0;
            loop {
                peak = peak.max(footprint().map_or(0, |f| f.now));
                // A stop (or a dropped sampler) ends it; the timeout is the sampling interval.
                if !matches!(stopped.recv_timeout(Duration::from_millis(1)), Err(mpsc::RecvTimeoutError::Timeout)) {
                    return peak.max(footprint().map_or(0, |f| f.now));
                }
            }
        });
        Self { stop, thread }
    }

    /// Stop and return the highest footprint seen, in bytes (0 if it could not be read).
    #[must_use]
    pub fn stop(self) -> u64 {
        let _ = self.stop.send(());
        self.thread.join().unwrap_or(0)
    }
}

/// Physical RAM (`hw.memsize`).
fn ram() -> u64 {
    let mut v = 0u64;
    let mut len = size_of::<u64>();
    // SAFETY: the name is NUL-terminated and `v` is a writable u64 of `len` bytes.
    let rc = unsafe {
        libc::sysctlbyname(c"hw.memsize".as_ptr(), (&raw mut v).cast::<c_void>(), &raw mut len, std::ptr::null_mut(), 0)
    };
    if rc == 0 { v } else { 0 }
}

/// The app's memory limit: what it uses now plus what iOS still allows.
#[cfg(target_os = "ios")]
fn app_limit() -> Option<u64> {
    unsafe extern "C" {
        /// Bytes the app can still allocate before iOS terminates it (`<os/proc.h>`).
        safe fn os_proc_available_memory() -> usize;
    }
    let now = footprint().map_or(0, |f| f.now);
    Some(now + os_proc_available_memory() as u64)
}

#[cfg(not(target_os = "ios"))]
fn app_limit() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::{PeakSampler, footprint};

    #[test]
    fn sampler_sees_a_freed_spike() {
        let before = footprint().unwrap().now;
        let s = PeakSampler::start();
        let spike = vec![1u8; 256 << 20];
        std::hint::black_box(&spike);
        // Long enough for several samples while the spike is resident.
        let t = std::time::Instant::now();
        while t.elapsed() < std::time::Duration::from_millis(20) {
            std::hint::spin_loop();
        }
        drop(spike);
        let peak = s.stop();
        assert!(peak >= before + (200 << 20), "peak {peak}, before {before}");
    }
}
