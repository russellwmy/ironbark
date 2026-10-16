// Tools the chat offers Maple, run in the app: a calculator (on device), the weather
// (Open-Meteo) and Wikipedia search. Neither service needs a key. Each result is a JSON string
// for a <tool_response>.

import Foundation

enum Tools {
    /// Function schemas for the system turn.
    static let schemas: [[String: Any]] = [
        tool("calculator", "Evaluate an arithmetic expression exactly: + - * / ^ % and parentheses.",
             "expression", "e.g. 2 * (3 + 4)"),
        tool("get_weather", "Current weather and a 3-day forecast for a city.", "city", "City name, e.g. Paris"),
        tool("wikipedia_search", "Search Wikipedia. Returns the intro of the top articles: facts about people, "
             + "places, events, science and history.", "query", "What to look up"),
    ]

    static var schemasJSON: String {
        let data = (try? JSONSerialization.data(withJSONObject: schemas, options: [.sortedKeys])) ?? Data("[]".utf8)
        return String(decoding: data, as: UTF8.self)
    }

    private static func tool(_ name: String, _ description: String, _ arg: String, _ argDescription: String) -> [String: Any] {
        [
            "type": "function",
            "function": [
                "name": name,
                "description": description,
                "parameters": [
                    "type": "object",
                    "properties": [arg: ["type": "string", "description": argDescription]],
                    "required": [arg],
                ],
            ],
        ]
    }

    /// Run a call (blocks on the network; not on the main queue).
    static func run(_ name: String, _ args: [String: Any]) -> String {
        let arg = { (k: String) in (args[k] as? String) ?? (args.values.first as? String) ?? "" }
        switch name {
        case "calculator": return calculator(arg("expression"))
        case "get_weather": return weather(arg("city"))
        case "wikipedia_search": return wikipedia(arg("query"))
        default: return json(["error": "no tool named \(name)"])
        }
    }

    static func json(_ v: Any) -> String {
        let data = (try? JSONSerialization.data(withJSONObject: v, options: [.sortedKeys, .withoutEscapingSlashes])) ?? Data()
        return String(decoding: data, as: UTF8.self)
    }

    // MARK: - Calculator

    static func calculator(_ expression: String) -> String {
        let e = expression.replacingOccurrences(of: "**", with: "^").replacingOccurrences(of: "×", with: "*")
            .replacingOccurrences(of: "÷", with: "/")
        var p = Arith(Array(e))
        guard let v = p.parse() else { return json(["error": "cannot evaluate \(expression)"]) }
        // Whole numbers print without a fraction when a Double holds them exactly.
        if v.rounded() == v, abs(v) < 9e15 { return json(["result": Int64(v)]) }
        return json(["result": v])
    }

    /// Recursive descent: sum = product (+|- product)*, product = power (*|/|% power)*,
    /// power = unary (^ power)?, unary = -unary | number | (sum).
    private struct Arith {
        let s: [Character]
        var i = 0
        init(_ s: [Character]) { self.s = s.filter { !$0.isWhitespace && $0 != "," } }

        mutating func parse() -> Double? {
            guard let v = sum(), i == s.count, v.isFinite else { return nil }
            return v
        }

        private mutating func eat(_ c: Character) -> Bool {
            if i < s.count, s[i] == c { i += 1; return true }
            return false
        }

        private mutating func sum() -> Double? {
            guard var v = product() else { return nil }
            while true {
                if eat("+") { guard let r = product() else { return nil }; v += r }
                else if eat("-") { guard let r = product() else { return nil }; v -= r }
                else { return v }
            }
        }

        private mutating func product() -> Double? {
            guard var v = power() else { return nil }
            while true {
                if eat("*") { guard let r = power() else { return nil }; v *= r }
                else if eat("/") { guard let r = power() else { return nil }; v /= r }
                else if eat("%") { guard let r = power() else { return nil }; v = v.truncatingRemainder(dividingBy: r) }
                else { return v }
            }
        }

        private mutating func power() -> Double? {
            guard let b = unary() else { return nil }
            if eat("^") {
                guard let e = power() else { return nil }
                return Foundation.pow(b, e)
            }
            return b
        }

        private mutating func unary() -> Double? {
            if eat("-") { return unary().map { -$0 } }
            if eat("+") { return unary() }
            if eat("(") {
                guard let v = sum(), eat(")") else { return nil }
                return v
            }
            let start = i
            while i < s.count, s[i].isNumber || s[i] == "." { i += 1 }
            return Double(String(s[start..<i]))
        }
    }

    // MARK: - Network

    /// GET `url` as JSON, waiting at most 15 s.
    private static func get(_ url: URL) -> Any? {
        var request = URLRequest(url: url, timeoutInterval: 15)
        // Wikimedia asks clients to name themselves.
        request.setValue("IronbarkBench/1.0 (dev.ironbark.bench)", forHTTPHeaderField: "User-Agent")
        let done = DispatchSemaphore(value: 0)
        var result: Any?
        URLSession.shared.dataTask(with: request) { data, _, _ in
            result = data.flatMap { try? JSONSerialization.jsonObject(with: $0) }
            done.signal()
        }.resume()
        _ = done.wait(timeout: .now() + 16)
        return result
    }

    private static func url(_ base: String, _ query: [String: String]) -> URL? {
        var c = URLComponents(string: base)
        c?.queryItems = query.sorted { $0.key < $1.key }.map { URLQueryItem(name: $0.key, value: $0.value) }
        return c?.url
    }

    // MARK: - Weather

    static func weather(_ city: String) -> String {
        guard let u = url("https://geocoding-api.open-meteo.com/v1/search", ["name": city, "count": "1"]),
              let geo = get(u) as? [String: Any], let place = (geo["results"] as? [[String: Any]])?.first,
              let lat = place["latitude"] as? Double, let lon = place["longitude"] as? Double
        else { return json(["error": "no place named \(city)"]) }
        let q = [
            "latitude": String(lat), "longitude": String(lon), "timezone": "auto", "forecast_days": "3",
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        ]
        guard let u = url("https://api.open-meteo.com/v1/forecast", q), let f = get(u) as? [String: Any],
              let now = f["current"] as? [String: Any]
        else { return json(["error": "weather service unavailable"]) }
        let code = { (v: Any?) in condition((v as? NSNumber)?.intValue ?? -1) }
        // One decimal: JSONSerialization writes a Double's 17 digits (28.899999999999999).
        let n = { (v: Any?) -> Any in
            (v as? NSNumber).map { Decimal(string: String(format: "%.1f", $0.doubleValue)) ?? 0 } ?? NSNull()
        }
        var days: [[String: Any]] = []
        if let d = f["daily"] as? [String: [Any]], let dates = d["time"] {
            for (k, date) in dates.enumerated() {
                days.append([
                    "date": date,
                    "condition": code(d["weather_code"]?[k]),
                    "max_c": n(d["temperature_2m_max"]?[k]),
                    "min_c": n(d["temperature_2m_min"]?[k]),
                    "rain_chance_pct": n(d["precipitation_probability_max"]?[k]),
                ])
            }
        }
        return json([
            "place": [place["name"], place["admin1"], place["country"]].compactMap { $0 as? String }.joined(separator: ", "),
            "local_time": now["time"] ?? NSNull(),
            "temp_c": n(now["temperature_2m"]),
            "feels_like_c": n(now["apparent_temperature"]),
            "humidity_pct": n(now["relative_humidity_2m"]),
            "wind_kmh": n(now["wind_speed_10m"]),
            "condition": code(now["weather_code"]),
            "forecast": days,
        ])
    }

    /// WMO weather interpretation codes.
    private static func condition(_ code: Int) -> String {
        switch code {
        case 0: "clear"
        case 1: "mainly clear"
        case 2: "partly cloudy"
        case 3: "overcast"
        case 45, 48: "fog"
        case 51, 53, 55, 56, 57: "drizzle"
        case 61, 63, 65, 66, 67: "rain"
        case 71, 73, 75, 77: "snow"
        case 80, 81, 82: "rain showers"
        case 85, 86: "snow showers"
        case 95, 96, 99: "thunderstorm"
        default: "unknown"
        }
    }

    // MARK: - Wikipedia

    static func wikipedia(_ query: String) -> String {
        let q = [
            "action": "query", "format": "json", "formatversion": "2", "generator": "search", "gsrsearch": query,
            "gsrlimit": "3", "prop": "extracts", "exintro": "1", "explaintext": "1", "exsentences": "5",
        ]
        guard let u = url("https://en.wikipedia.org/w/api.php", q), let r = get(u) as? [String: Any]
        else { return json(["error": "Wikipedia unavailable"]) }
        let pages = ((r["query"] as? [String: Any])?["pages"] as? [[String: Any]] ?? [])
            .sorted { ($0["index"] as? Int ?? 0) < ($1["index"] as? Int ?? 0) }
            .map { p -> [String: Any] in
                let title = p["title"] as? String ?? ""
                return [
                    "title": title,
                    "url": "https://en.wikipedia.org/wiki/" + title.replacingOccurrences(of: " ", with: "_"),
                    "extract": p["extract"] as? String ?? "",
                ]
            }
        return pages.isEmpty ? json(["error": "no results for \(query)"]) : json(pages)
    }
}
