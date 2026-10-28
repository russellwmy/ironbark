// IronbarkBench: chat with Qwen3.6 or Maple and benchmark them on iPhone, through the C API in
// ironbark.h. Models are `ironbark pack` dirs in Documents (default: a Qwen pack, else the first). The chat can call the
// tools in Tools.swift, renders replies with Markdown.swift, and (Qwen) takes a photo with a message.
//
// Benchmark without the UI: put arguments in Documents/auto.txt (or pass them at launch with
// -auto); the app runs them at its next launch, deletes the file, and writes one JSON line per
// result to Documents/results.jsonl, ending with IRONBARK_DONE (IRONBARK_CANCELLED if the app
// left the screen).
//   -models maple,maple-2bit (default: as the chat)  -budgets 0,2,1.5 (GB; 0 = from the device)  -tokens 128
//   -chars 3000  -profile  -prompts prose,rust
//   -lookahead 1,2 (each budget at each lookahead; default 1)  -flash (each also with FlashHead)
//   -dynamic 0,0.08,0.10 (each with dynamic top-k off, then dropping experts under each share;
//   default 0)
//   -chat (instead of the raw prompts: a scripted 4-turn conversation per model, budget and
//   -dynamic setting, one line per turn; -tokens caps each reply)
//   -minutes 30 (with -chat: repeat the conversation for 30 minutes per setting, each line with
//   the battery level and thermal state: energy over the same work; unplug the phone)
//   -dynamic default (each model at the app's Expert drop)
//   -tools 1024 (tool-calling cases after the runs, replies of at most 1024 tokens; -budgets ""
//   skips the runs)
//   -delete maple,maple-2bit (remove these model dirs first: devicectl cannot delete files)
//   -uichat 6 (send 6 questions through the real chat, timing the main thread)
//   -uistress 10 (no model: replay 10 tool-chat turns into the chat view, timing the main thread)
//   -image cats.jpg -ask "Describe this photo." (Qwen: each model and budget answers about an image
//   in Documents instead of the runs; -pixels 262144 caps its resized size, -tokens the reply)

import BackgroundTasks
import PhotosUI
import SwiftUI

@main
struct IronbarkBenchApp: App {
    @StateObject private var chat = Chat()
    @StateObject private var bench = Bench()
    @Environment(\.scenePhase) private var phase

    var body: some Scene {
        WindowGroup {
            TabView {
                ChatView(chat: chat).tabItem { Label("Chat", systemImage: "bubble.left.and.bubble.right") }
                BenchView(bench: bench).tabItem { Label("Bench", systemImage: "speedometer") }
            }
            .onAppear {
                UIApplication.shared.isIdleTimerDisabled = true
                StallLog.start()
                // One model in memory at a time: the bench frees the chat's, and the chat waits.
                bench.freeOthers = chat.free
                bench.stressChat = chat.stress
                bench.realChat = chat.stressReal
                chat.otherBusy = { bench.busy }
                if CommandLine.arguments.contains("-auto") || bench.loadAutoFile() { bench.auto() }
            }
            .onChange(of: phase) { _, p in
                // iOS allows no GPU work off screen: stop the bench and the reply (unless it runs as a
                // continued-processing task); the chat frees its model after `keepOffScreen` seconds.
                chat.onScreen(p == .active)
                if p != .active { bench.cancel = true }
            }
        }
    }
}

// MARK: - Library

let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]

/// A JSON string from the library as a dictionary, freed after reading.
func take(_ p: UnsafeMutablePointer<CChar>?) -> [String: Any] {
    guard let p else { return ["error": "null"] }
    defer { ironbark_free_string(p) }
    let data = Data(String(cString: p).utf8)
    return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? ["error": "bad json"]
}

/// Open `model` (a dir in Documents) within `budget` GB of footprint (0: from the device).
func open(model: String, budget: Double) -> (OpaquePointer?, String?) {
    var err: UnsafeMutablePointer<CChar>?
    let path = docs.appendingPathComponent(model).path
    let s = ironbark_open(path, UInt64(budget * 1e9), 0, &err)
    if s == nil {
        let msg = err.map { String(cString: $0) } ?? "open failed"
        if let err { ironbark_free_string(err) }
        return (nil, msg)
    }
    return (s, nil)
}

/// Pack dirs in Documents (they hold experts.json).
func models() -> [String] {
    let names = (try? FileManager.default.contentsOfDirectory(atPath: docs.path)) ?? []
    return names.filter { FileManager.default.fileExists(atPath: docs.appendingPathComponent("\($0)/experts.json").path) }.sorted()
}

/// RGB bytes of `image` (its orientation applied, rows top to bottom), scaled down first to at
/// most `pixels` pixels (the library resizes it again to the model's grid).
func rgb(_ image: UIImage, pixels: Int) -> (bytes: [UInt8], width: Int, height: Int)? {
    let (w0, h0) = (image.size.width * image.scale, image.size.height * image.scale)
    guard w0 >= 1, h0 >= 1 else { return nil }
    let s = min(1, (Double(pixels) / Double(w0 * h0)).squareRoot())
    let (w, h) = (max(1, Int(w0 * s)), max(1, Int(h0 * s)))
    var rgba = [UInt8](repeating: 0, count: w * h * 4)
    let drawn = rgba.withUnsafeMutableBytes { buf -> Bool in
        guard let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return false }
        ctx.interpolationQuality = .high
        // UIKit coordinates (origin top left), so the first row in memory is the image's top.
        ctx.translateBy(x: 0, y: CGFloat(h))
        ctx.scaleBy(x: 1, y: -1)
        UIGraphicsPushContext(ctx)
        image.draw(in: CGRect(x: 0, y: 0, width: w, height: h))
        UIGraphicsPopContext()
        return true
    }
    guard drawn else { return nil }
    var out = [UInt8](repeating: 0, count: w * h * 3)
    for i in 0..<(w * h) {
        out[3 * i] = rgba[4 * i]
        out[3 * i + 1] = rgba[4 * i + 1]
        out[3 * i + 2] = rgba[4 * i + 2]
    }
    return (out, w, h)
}

/// Encode `image` for the session's next message (Qwen); the library's JSON reply.
func setImage(_ s: OpaquePointer, _ image: UIImage, maxPixels: Int) -> [String: Any] {
    let limit = maxPixels == 0 ? 512 * 512 : maxPixels
    guard let px = rgb(image, pixels: 2 * limit) else { return ["error": "could not read the image"] }
    return px.bytes.withUnsafeBufferPointer {
        take(ironbark_set_image(s, $0.baseAddress, UInt32(px.width), UInt32(px.height), UInt32(maxPixels)))
    }
}

/// The model used unless one is chosen: a Qwen pack if there is one, else the first pack.
func defaultModel() -> String {
    let all = models()
    return all.first { $0.hasPrefix("qwen") } ?? all.first ?? "maple"
}

/// The Expert drop a model starts with, from the evals: Qwen 0.08 (0.10: full IFEval −1.5),
/// others off until measured.
func defaultDrop(_ model: String) -> Double {
    if model.hasPrefix("qwen") { return 0.08 }
    return 0
}

func num(_ d: [String: Any], _ keys: String...) -> Double {
    var v: Any? = d
    for k in keys { v = (v as? [String: Any])?[k] }
    return (v as? NSNumber)?.doubleValue ?? .nan
}

// MARK: - Chat

struct Message: Identifiable {
    enum Role: String { case user, assistant, tool }
    let id = UUID()
    let role: Role
    /// What the user wrote, or the reply as streamed (reasoning, answer, tool calls).
    var text = ""
    /// Tool messages: each call ("name(args)") and its result.
    var calls: [(call: String, result: String)] = []
    /// A photo sent with a user message (Qwen reads it with this message only).
    var image: UIImage?

    /// As ironbark_send reads it.
    var json: [String: Any] {
        ["role": role.rawValue, "content": role == .tool ? calls.map(\.result) as Any : text]
    }
}

/// A streamed reply split for display: reasoning, answer, and whether it calls a tool. The
/// prompt opens the reasoning, so a reply starts inside it (as ironbark::parse).
struct Shown {
    var thinking = "", answer = "", calling = false, thought = false

    init(_ text: String) {
        let raw = text
        var rest = Substring(raw)
        if let end = raw.range(of: "</think>") {
            thinking = String(raw[..<end.lowerBound])
            rest = raw[end.upperBound...]
            thought = true
        } else if !raw.contains("<tool_call>") {
            thinking = raw
            rest = ""
        }
        // Later reasoning (a model may open another, often empty, after a tool result) is reasoning too.
        while let open = rest.range(of: "<think>") {
            let end = rest.range(of: "</think>", range: open.upperBound..<rest.endIndex)
            thinking += "\n" + rest[open.upperBound..<(end?.lowerBound ?? rest.endIndex)]
            rest = rest[..<open.lowerBound] + (end.map { rest[$0.upperBound...] } ?? "")
        }
        thinking = thinking.replacingOccurrences(of: "<think>", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
        while let open = rest.range(of: "<tool_call>") {
            answer += rest[..<open.lowerBound]
            calling = true
            rest = rest.range(of: "</tool_call>", range: open.upperBound..<rest.endIndex).map { rest[$0.upperBound...] } ?? ""
        }
        answer = (answer + rest).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

final class Chat: ObservableObject {
    @Published var messages: [Message] = []
    /// The user is dragging the chat or it is still coasting (main queue): reply text waits in
    /// `heldText`, as a row growing under a moving scroll view sent SwiftUI into an endless
    /// update cycle (the app froze, then was killed).
    var scrolling = false {
        didSet { if !scrolling { releaseText() } }
    }
    private var heldText = ""

    /// Add streamed text to the last message, or hold it while the user scrolls (main queue).
    func stream(_ text: String) {
        heldText += text
        if !scrolling { releaseText() }
    }

    /// Show held text, e.g. before a new message is added (main queue).
    func releaseText() {
        guard !heldText.isEmpty, !messages.isEmpty else { return }
        messages[messages.count - 1].text += heldText
        heldText = ""
    }
    @Published var busy = false
    @Published var status = ""
    /// Choosing a model resets Expert drop to that model's default.
    @Published var model = defaultModel() {
        didSet { if model != oldValue { drop = defaultDrop(model) } }
    }
    // 4.5 GB: more expert slots per layer than the device default (3 GB).
    @Published var budget = 4.5
    /// Reason before answering a message (Maple's replies to tool results always do).
    @Published var think = false
    /// Dynamic top-k: drop each token's experts under this share of its routing (0: off); per
    /// model, [`defaultDrop`].
    @Published var drop = defaultDrop(defaultModel())

    /// Every library call runs here, one at a time.
    private let engine = DispatchQueue(label: "ironbark.engine", qos: .userInitiated)
    // Engine queue only:
    private var session: OpaquePointer?
    private var opened: (String, Double)?
    /// Messages of the transcript the session's KV cache holds; otherwise the next turn
    /// rebuilds the conversation from the transcript.
    private var cached = 0
    /// Oldest messages of the transcript dropped to fit the context: neither sent nor cached.
    private var forgotten = 0
    /// The reply loop stops at its next token.
    private var stop = false
    /// The app is off screen: no GPU work unless iOS granted a continued-processing task.
    private var offScreen = false
    private var inTask = false

    static let taskPrefix = "dev.ironbark.bench.reply"
    private var registered = false
    /// A submission was refused (e.g. no Background GPU Access entitlement): do not retry.
    private var refused = false
    private var pending: ((AnyObject?) -> Void)?

    /// Longest reply (reasoning included) per round, and rounds of tool calls per message.
    static let maxTokens = 2048
    static let maxRounds = 3

    /// Something else holds a model (the bench); main queue.
    var otherBusy: () -> Bool = { false }

    /// Start a reply; false (nothing sent) while the bench runs.
    @discardableResult
    func send(_ question: String, image: UIImage? = nil) -> Bool {
        if otherBusy() {
            status = "The bench is running: send after it finishes"
            return false
        }
        let history = messages
        messages.append(Message(role: .user, text: question, image: image))
        let (model, budget, think, drop) = (self.model, self.budget, self.think, self.drop)
        busy = true
        stop = false
        let job: (AnyObject?) -> Void = { task in
            self.reply(question, image, history, model, budget, think, drop, task)
        }
        if !continueInBackground(job) {
            engine.async { job(nil) }
        }
        return true
    }

    /// iOS 26: run the reply as a continued-processing task, so it can finish on the GPU if the
    /// app leaves the screen. False when the device or the app is not allowed to.
    func continueInBackground(_ job: @escaping (AnyObject?) -> Void) -> Bool {
        guard #available(iOS 26, *), !refused, BGTaskScheduler.supportedResources.contains(.gpu) else { return false }
        if !registered {
            registered = BGTaskScheduler.shared.register(forTaskWithIdentifier: "\(Self.taskPrefix).*", using: engine) { task in
                guard let job = self.pending else {
                    task.setTaskCompleted(success: false)
                    return
                }
                self.pending = nil
                task.expirationHandler = { self.stop = true }
                job(task)
            }
            if !registered { return false }
        }
        pending = job
        let request = BGContinuedProcessingTaskRequest(
            identifier: "\(Self.taskPrefix).\(UUID().uuidString)", title: "Maple", subtitle: "Writing a reply")
        request.requiredResources = .gpu
        request.strategy = .fail
        do {
            try BGTaskScheduler.shared.submit(request)
            return true
        } catch {
            pending = nil
            refused = true
            DispatchQueue.main.async { self.status = "Replies pause off screen (no background GPU): \(error.localizedDescription)" }
            return false
        }
    }

    /// One turn, on the engine queue. `task` is a `BGContinuedProcessingTask` or nil.
    private func reply(
        _ question: String, _ image: UIImage?, _ history: [Message], _ model: String, _ budget: Double, _ think: Bool,
        _ drop: Double, _ task: AnyObject?
    ) {
        inTask = task != nil
        defer { inTask = false }
        if #available(iOS 26, *), let t = task as? BGContinuedProcessingTask {
            t.progress.totalUnitCount = Int64(Self.maxTokens)
        }
        if opened.map({ $0 != (model, budget) }) ?? true {
            close()
            let (s, err) = open(model: model, budget: budget)
            session = s
            opened = s == nil ? nil : (model, budget)
            if let err { finish("Error: \(err)", task); return }
            if let s {
                let date = Date().formatted(.dateTime.weekday(.wide).year().month(.wide).day())
                let iso = Date().formatted(.iso8601.year().month().day())
                // Tools only when needed: each call's result is read in full before the reply.
                let system = "You are a helpful assistant. Today is \(date) (\(iso)). Answer in Markdown when it helps. "
                    + "Use a tool only for live data, facts you do not know, exact arithmetic or counting days."
                _ = take(ironbark_set_system(s, system, Tools.schemasJSON))
            }
        }
        guard let s = session else { return }
        ironbark_set_thinking(s, think)
        ironbark_set_dynamic_k(s, Float(drop), 3)
        // The photo goes with this message: encoded now, read with the message below.
        var seen = ""
        if let image {
            setStatus("Reading the photo…")
            let r = setImage(s, image, maxPixels: 0)
            if let e = r["error"] { finish("Error: \(e)", task); return }
            seen = String(format: "photo %.0f×%.0f → %.0f tokens in %.1f s · ", num(r, "width"), num(r, "height"),
                          num(r, "tokens"), num(r, "encode_s"))
        }
        // Continue the conversation in the KV cache, or rebuild it from the transcript.
        var history = Array(history.dropFirst(min(forgotten, history.count)))
        var rebuild = cached != history.count
        var send = (rebuild ? history : []).map(\.json) + [Message(role: .user, text: question).json]
        var held = history.count + 1
        var dropped = 0
        var count = 0
        /// Why the last reply stopped before the model ended its turn, if it did.
        var cut = ""
        /// Results of this message's calls by "name(args)": a repeated call is not run again.
        var done: [String: String] = [:]
        /// The next round may not call tools (a repeated call, or the round limit).
        var answerNow = false
        // One round past the limit, with tool calls off: the model answers with what it has.
        for round in 0...Self.maxRounds + 1 {
            ironbark_set_tool_calls(s, !answerNow && round <= Self.maxRounds)
            var start = take(ironbark_send(s, Tools.json(send), rebuild && round == 0))
            // The message does not fit: forget the oldest exchanges until it does.
            while round == 0, (start["error"] as? String)?.hasPrefix("context full") == true, !history.isEmpty {
                let next = history.dropFirst().firstIndex { $0.role == .user } ?? history.endIndex
                dropped += next
                forgotten += next
                history.removeFirst(next)
                rebuild = true
                send = history.map(\.json) + [Message(role: .user, text: question).json]
                held = history.count + 1
                start = take(ironbark_send(s, Tools.json(send), true))
            }
            if let e = start["error"] { cached = 0; forgotten = 0; finish("Error: \(e)", task); return }
            DispatchQueue.main.async { self.releaseText(); self.messages.append(Message(role: .assistant)) }
            var raw = ""
            var ended = false
            var tokens = 0
            // Pieces not shown yet: the UI takes them at most every 0.1 s (each update redraws
            // the reply and parses its Markdown).
            var unshown = ""
            var shownAt = Date()
            func show() {
                guard !unshown.isEmpty else { return }
                let text = unshown
                unshown = ""
                shownAt = Date()
                DispatchQueue.main.async { self.stream(text) }
            }
            while !stop {
                if offScreen && task == nil { break } // no GPU off screen without a task
                guard let p = ironbark_next(s) else { ended = true; break }
                let piece = String(cString: p)
                ironbark_free_string(p)
                raw += piece
                count += 1
                tokens += 1
                if #available(iOS 26, *), let t = task as? BGContinuedProcessingTask {
                    t.progress.completedUnitCount = Int64(min(count, Self.maxTokens))
                }
                unshown += piece
                if Date().timeIntervalSince(shownAt) >= 0.1 { show() }
                if tokens >= Self.maxTokens { break }
            }
            show()
            // ironbark_next also returns NULL on an error: a reply that ended without a reason
            // failed (e.g. a GPU step off screen; the session reads the conversation again and
            // goes on with the next message).
            if ended, take(ironbark_stats(s))["finish"] is NSNull {
                cut = "the GPU step failed"
                held += 1
                break
            }
            if stop {
                cut = "stopped"
            } else if !ended {
                cut = tokens >= Self.maxTokens ? "reply limit (\(Self.maxTokens) tokens)" : "paused off screen"
            }
            held += 1
            // The session closes a reply cut short, so the KV cache still matches the transcript.
            let calls = ended ? take(ironbark_parse(raw))["calls"] as? [[String: Any]] ?? [] : []
            if calls.isEmpty || round == Self.maxRounds + 1 || stop { break }
            let last = round == Self.maxRounds
            if !last { setStatus("Running \(calls.compactMap { $0["name"] as? String }.joined(separator: ", "))…") }
            var tool = Message(role: .tool)
            for c in calls {
                let name = c["name"] as? String ?? ""
                let args = c["arguments"] as? [String: Any] ?? [:]
                let shown = args.sorted { $0.key < $1.key }.map { "\($0.key): \($0.value)" }.joined(separator: ", ")
                let key = "\(name)(\(shown))"
                let result: String
                if let earlier = done[key] {
                    // The model asked again for a result it has: give it back and make it answer.
                    result = Tools.json(["note": "You already called this; same result. Answer now with what you have.",
                                         "result": earlier])
                    answerNow = true
                } else if last {
                    result = Tools.json(["error": "No more tool calls for this message. Answer now with what you have."])
                } else {
                    result = Tools.run(name, args)
                    done[key] = result
                }
                tool.calls.append((key, result))
            }
            DispatchQueue.main.async { self.releaseText(); self.messages.append(tool) }
            held += 1
            send = [tool.json]
        }
        cached = held
        let st = take(ironbark_stats(s))
        if cut.isEmpty, st["finish"] as? String == "context" { cut = "context full" }
        let forgot = dropped > 0 ? "Forgot the \(dropped) oldest messages (context full) · " : ""
        let line = seen + forgot + (cut.isEmpty ? "" : "Cut short: \(cut) · ") + String(
            format: "%.1f tok/s · prompt %.0f tok/s · context %.0f · %.0f experts/layer · hit %.0f%% · footprint %.2f GB (peak %.2f)",
            num(st, "decode_tps"), num(st, "prefill_tps"), num(st, "context"), num(st, "experts", "per_layer"),
            100 * num(st, "experts", "hit"), num(st, "memory", "footprint") / 1e9, num(st, "memory", "footprint_peak") / 1e9)
        finish(line, task)
        if offScreen { close() }
    }

    /// Free the model (engine queue).
    private func close() {
        if let s = session { ironbark_close(s) }
        session = nil
        opened = nil
        cached = 0
    }

    /// Seconds the model stays loaded off screen, so a quick switch away needs no reload.
    static let keepOffScreen = 20.0
    /// Main queue: bumped on each screen change, so a pending close knows the app came back.
    private var screenChanges = 0
    private var keepTask = UIBackgroundTaskIdentifier.invalid

    /// The app left the screen or came back. Off screen, a reply without a background task stops
    /// at its next token; the model is freed after `keepOffScreen` seconds (it reloads on the
    /// next message) unless the app is back by then.
    func onScreen(_ visible: Bool) {
        offScreen = !visible
        screenChanges += 1
        if visible { endKeep(); return }
        if !inTask { stop = true }
        let change = screenChanges
        // A background task keeps the app running for the wait (iOS allows about 30 s).
        if keepTask == .invalid {
            keepTask = UIApplication.shared.beginBackgroundTask(withName: "Free the model") { self.closeOffScreen(change, now: true) }
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + Self.keepOffScreen) { self.closeOffScreen(change, now: false) }
    }

    /// Main queue: free the model if the app is still off screen since `change`.
    private func closeOffScreen(_ change: Int, now: Bool) {
        if change == screenChanges {
            engine.async { if !self.inTask { self.close() } }
        }
        // On expiry the task must end now; the close above still runs.
        if now || change == screenChanges { endKeep() }
    }

    private func endKeep() {
        if keepTask != .invalid {
            UIApplication.shared.endBackgroundTask(keepTask)
            keepTask = .invalid
        }
    }

    func cancel() { stop = true }

    /// Stop any reply and free the model now (not on the main queue; waits for the reply to
    /// stop at its next token). The next message reopens it.
    func free() {
        stop = true
        engine.sync { close() }
    }

    func reset() {
        heldText = ""
        messages = []
        status = ""
        engine.async { self.cached = 0; self.forgotten = 0 }
    }

    private func setStatus(_ line: String) { DispatchQueue.main.async { self.status = line } }

    private func finish(_ line: String, _ task: AnyObject?) {
        if #available(iOS 26, *), let t = task as? BGContinuedProcessingTask {
            t.setTaskCompleted(success: true)
        }
        DispatchQueue.main.async {
            self.status = line
            self.busy = false
        }
    }
}

/// Equatable: a reply streaming redraws only itself, not every message above it.
struct MessageView: View, Equatable {
    let m: Message

    static func == (a: Self, b: Self) -> Bool {
        a.m.id == b.m.id && a.m.text == b.m.text && a.m.calls.count == b.m.calls.count
    }

    var body: some View {
        switch m.role {
        case .user:
            VStack(alignment: .trailing, spacing: 6) {
                if let img = m.image {
                    Image(uiImage: img).resizable().scaledToFit().frame(maxWidth: 220, maxHeight: 220)
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                }
                Text(m.text)
            }
                .padding(10)
                .background(Color.accentColor.opacity(0.15))
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .textSelection(.enabled)
        case .tool:
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(m.calls.enumerated()), id: \.offset) { _, c in
                    DisclosureGroup {
                        Text(c.result).font(.caption2.monospaced()).textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    } label: {
                        Label(c.call, systemImage: "wrench.and.screwdriver").font(.caption).lineLimit(2)
                    }
                }
            }
            .padding(10)
            .background(Color.secondary.opacity(0.06))
            .clipShape(RoundedRectangle(cornerRadius: 12))
        case .assistant:
            let r = Shown(m.text)
            VStack(alignment: .leading, spacing: 8) {
                if !r.thinking.isEmpty {
                    DisclosureGroup {
                        Text(r.thinking).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    } label: {
                        Label(r.thought ? "Thought" : "Thinking…", systemImage: "brain").font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                if !r.answer.isEmpty {
                    MarkdownView(text: r.answer).textSelection(.enabled)
                } else if r.thinking.isEmpty && !r.calling {
                    Text("…")
                }
                if r.calling {
                    Label("Calling a tool", systemImage: "wrench.and.screwdriver").font(.caption).foregroundStyle(.secondary)
                }
            }
            .padding(10)
            .background(Color.secondary.opacity(0.1))
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }
    }
}

/// The system camera: hands the photo taken to `took`, then closes.
struct Camera: UIViewControllerRepresentable {
    let took: (UIImage) -> Void
    @Environment(\.dismiss) private var dismiss

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let c = UIImagePickerController()
        c.sourceType = .camera
        c.delegate = context.coordinator
        return c
    }

    func updateUIViewController(_ c: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    final class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: Camera
        init(_ parent: Camera) { self.parent = parent }

        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            if let img = info[.originalImage] as? UIImage { parent.took(img) }
            parent.dismiss()
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) { parent.dismiss() }
    }
}

struct ChatView: View {
    @ObservedObject var chat: Chat
    @State private var input = ""
    @FocusState private var typing: Bool
    /// A photo to send with the next message.
    @State private var picked: PhotosPickerItem?
    @State private var photo: UIImage?
    @State private var camera = false
    /// Keep the newest text in view; off while the user scrolls up, on again at the bottom.
    @State private var following = true

    var body: some View {
        NavigationStack {
            VStack(spacing: 8) {
                ScrollViewReader { proxy in
                    ScrollView {
                        // Not lazy: a LazyVStack re-estimates row heights as the last row grows.
                        VStack(alignment: .leading, spacing: 10) {
                            ForEach(chat.messages) { m in
                                MessageView(m: m).equatable()
                                    .frame(maxWidth: .infinity, alignment: m.role == .user ? .trailing : .leading)
                                    .id(m.id)
                            }
                        }
                        .padding(.horizontal)
                    }
                    // Scrolling or tapping the messages hides the keyboard.
                    .scrollDismissesKeyboard(.interactively)
                    .onTapGesture { typing = false }
                    .onScrollPhaseChange { _, phase, context in
                        StallLog.note(["scroll": "\(phase)", "following": following])
                        chat.scrolling = phase != .idle
                        if phase == .interacting {
                            following = false
                        } else if phase == .idle {
                            let g = context.geometry
                            following = g.contentOffset.y + g.containerSize.height >= g.contentSize.height - 80
                        }
                    }
                    .onChange(of: chat.messages.count) { _, n in
                        StallLog.note(["messages": n, "chars": chat.messages.reduce(0) { $0 + $1.text.count },
                                       "busy": chat.busy, "last": "message added"])
                        // A message the user sent: follow it, wherever the view was.
                        if chat.messages.last?.role == .user { following = true }
                        if following, let id = chat.messages.last?.id { proxy.scrollTo(id, anchor: .bottom) }
                    }
                    .onChange(of: chat.messages.last?.text) { _, text in
                        StallLog.note(["last_chars": text?.count ?? 0, "busy": chat.busy, "last": "text streamed"])
                        if following, let id = chat.messages.last?.id { proxy.scrollTo(id, anchor: .bottom) }
                    }
                }
                if !chat.status.isEmpty {
                    Text(chat.status).font(.caption2).foregroundStyle(.secondary).padding(.horizontal)
                }
                if let photo {
                    HStack {
                        Image(uiImage: photo).resizable().scaledToFill().frame(width: 56, height: 56)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        Button { self.photo = nil; picked = nil } label: { Image(systemName: "xmark.circle.fill") }
                        Spacer()
                    }
                    .padding(.horizontal)
                }
                HStack {
                    if UIImagePickerController.isSourceTypeAvailable(.camera) {
                        Button { camera = true } label: { Image(systemName: "camera") }.disabled(chat.busy)
                    }
                    PhotosPicker(selection: $picked, matching: .images) { Image(systemName: "photo") }
                        .disabled(chat.busy)
                    TextField("Message", text: $input, axis: .vertical).textFieldStyle(.roundedBorder)
                        .focused($typing)
                    if chat.busy {
                        Button("Stop") { chat.cancel() }
                    } else {
                        Button("Send") {
                            var q = input.trimmingCharacters(in: .whitespacesAndNewlines)
                            if q.isEmpty && photo != nil { q = "Describe this photo." }
                            guard !q.isEmpty, chat.send(q, image: photo) else { return }
                            input = ""
                            photo = nil
                            picked = nil
                            typing = false
                        }
                    }
                }
                .padding([.horizontal, .bottom])
                .fullScreenCover(isPresented: $camera) {
                    Camera { img in photo = img }.ignoresSafeArea()
                }
                .onChange(of: picked) { _, item in
                    guard let item else { return }
                    Task {
                        if let data = try? await item.loadTransferable(type: Data.self), let img = UIImage(data: data) {
                            photo = img
                        }
                    }
                }
            }
            .navigationTitle(chat.model)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Menu {
                        Picker("Model", selection: $chat.model) { ForEach(models(), id: \.self) { Text($0) } }
                        Toggle("Think", isOn: $chat.think)
                        Picker("Expert drop", selection: $chat.drop) {
                            ForEach([0.0, 0.08, 0.10], id: \.self) { Text($0 == 0 ? "Off" : String(format: "%.2f", $0)) }
                        }
                        Picker("Budget", selection: $chat.budget) {
                            ForEach([0.0, 1.5, 2.0, 3.0, 4.5], id: \.self) { Text($0 == 0 ? "Auto" : String(format: "%.1f GB", $0)) }
                        }
                    } label: { Image(systemName: "slider.horizontal.3") }
                    .disabled(chat.busy)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("New") { chat.reset() }.disabled(chat.busy)
                }
            }
        }
    }
}

// MARK: - Bench

struct Row: Identifiable {
    let id = UUID()
    let model: String, budget: Double, prompt: String
    let r: [String: Any]
}

struct ToolRow: Identifiable {
    let id = UUID()
    let r: [String: Any]
}

struct Part: Identifiable {
    let id = UUID()
    let model: String, name: String, us: Double, perToken: Double
}

final class Bench: ObservableObject {
    @Published var rows: [Row] = []
    @Published var parts: [Part] = []
    @Published var tools: [ToolRow] = []
    @Published var probe: [String: Any] = [:]
    @Published var busy = false
    @Published var log = ""
    var fileArgs: [String]?
    /// Frees other models (the chat's) before a run, so only the bench's counts.
    var freeOthers: () -> Void = {}
    /// Replays chat turns without the model (`-uistress`; see UIStress.swift).
    var stressChat: (Int, String?, Bool, @escaping ([String: Any]) -> Void, @escaping () -> Void) -> Void = { _, _, _, _, done in done() }
    /// Sends questions through the real chat (`-uichat`; see UIStress.swift).
    var realChat: (Int, @escaping ([String: Any]) -> Void, @escaping () -> Void) -> Void = { _, _, done in done() }
    /// Set when the app leaves the screen: the run stops at its next step.
    var cancel = false

    /// Read and delete Documents/auto.txt; whether it was there.
    func loadAutoFile() -> Bool {
        let url = docs.appendingPathComponent("auto.txt")
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return false }
        try? FileManager.default.removeItem(at: url)
        fileArgs = text.split(whereSeparator: \.isWhitespace).map(String.init)
        return true
    }

    /// Value after `name` in auto.txt or the launch arguments.
    func arg(_ name: String) -> String? {
        let a = fileArgs ?? CommandLine.arguments
        guard let i = a.firstIndex(of: name), i + 1 < a.count else { return nil }
        return a[i + 1]
    }

    func flag(_ name: String) -> Bool { (fileArgs ?? CommandLine.arguments).contains(name) }

    func emit(_ obj: [String: Any]) {
        let line = (try? JSONSerialization.data(withJSONObject: obj)).map { String(decoding: $0, as: UTF8.self) } ?? "{}"
        append(line)
    }

    func append(_ line: String) {
        print(line)
        let url = docs.appendingPathComponent("results.jsonl")
        if let h = try? FileHandle(forWritingTo: url) {
            h.seekToEndOfFile()
            h.write(Data((line + "\n").utf8))
            try? h.close()
        } else {
            try? (line + "\n").write(to: url, atomically: true, encoding: .utf8)
        }
    }

    func prompt(_ name: String, chars: Int) -> String {
        guard let url = Bundle.main.url(forResource: name, withExtension: "txt", subdirectory: "prompts"),
              let text = try? String(contentsOf: url, encoding: .utf8) else { return "" }
        return String(text.prefix(chars))
    }

    func run(models: [String], budgets: [Double], tokens: UInt32, chars: Int, profile: Bool, toolTokens: UInt32 = 0) {
        busy = true
        cancel = false
        rows = []
        parts = []
        tools = []
        try? FileManager.default.removeItem(at: docs.appendingPathComponent("results.jsonl"))
        DispatchQueue.global(qos: .userInitiated).async {
            self.freeOthers()
            let p = take(ironbark_probe(docs.appendingPathComponent(models.first ?? "maple").path))
            self.emit(["probe": p])
            DispatchQueue.main.async { self.probe = p }
            for model in models {
                for (i, b) in budgets.enumerated() where !self.cancel {
                    self.say("\(model) at \(b == 0 ? "auto" : "\(b) GB")…")
                    let (s, err) = open(model: model, budget: b)
                    guard let s else { self.emit(["model": model, "budget_gb": b, "error": err ?? ""]); continue }
                    if profile && i == 0 {
                        let pr = take(ironbark_profile(s))
                        self.emit(["model": model, "profile": pr])
                        let ps = (pr["parts"] as? [[String: Any]] ?? []).map {
                            Part(model: model, name: $0["part"] as? String ?? "", us: num($0, "us"), perToken: num($0, "per_token"))
                        }
                        DispatchQueue.main.async { self.parts += ps }
                    }
                    _ = take(ironbark_generate(s, "Hello", 4)) // compile the kernels
                    let prompts = (self.arg("-prompts") ?? "prose,rust").split(separator: ",").map(String.init)
                    let lookaheads = (self.arg("-lookahead") ?? "1").split(separator: ",").compactMap { UInt32($0) }
                    let flashes = self.flag("-flash") ? [false, true] : [false]
                    let drops = (self.arg("-dynamic") ?? "0").split(separator: ",").compactMap { Float($0) }
                    for drop in drops {
                        ironbark_set_dynamic_k(s, drop, 3)
                        for la in lookaheads {
                            ironbark_set_lookahead(s, la)
                            for flash in flashes {
                                _ = take(ironbark_set_flash_head(s, flash))
                                for name in prompts where !self.cancel {
                                    let r = take(ironbark_generate(s, self.prompt(name, chars: chars), tokens))
                                    self.emit(["model": model, "budget_gb": b, "prompt": name, "lookahead": la,
                                               "flash": flash, "drop": drop, "result": r])
                                    DispatchQueue.main.async {
                                        self.rows.append(Row(model: model, budget: b, prompt: name, r: r))
                                    }
                                }
                            }
                        }
                    }
                    ironbark_close(s)
                }
            }
            if toolTokens > 0 && !self.cancel, let model = models.first {
                self.toolCheck(model: model, budget: budgets.first ?? 0, maxTokens: toolTokens)
            }
            self.append(self.cancel ? "IRONBARK_CANCELLED" : "IRONBARK_DONE")
            self.say(self.cancel ? "Stopped: the app left the screen" : "Done")
            DispatchQueue.main.async { self.busy = false }
        }
    }

    /// Qwen: each model at each budget answers `question` about the image `file` in Documents, at
    /// most `tokens` tokens (reasoning off); one result line each.
    func runImage(models: [String], budgets: [Double], file: String, question: String, tokens: Int, maxPixels: Int) {
        busy = true
        cancel = false
        try? FileManager.default.removeItem(at: docs.appendingPathComponent("results.jsonl"))
        DispatchQueue.global(qos: .userInitiated).async {
            self.freeOthers()
            guard let image = UIImage(contentsOfFile: docs.appendingPathComponent(file).path) else {
                self.emit(["image": file, "error": "no such image in Documents"])
                self.append("IRONBARK_DONE")
                DispatchQueue.main.async { self.busy = false }
                return
            }
            for model in models {
                for b in budgets where !self.cancel {
                    self.say("\(model) at \(b == 0 ? "auto" : "\(b) GB"): \(file)…")
                    let (s, err) = open(model: model, budget: b)
                    guard let s else { self.emit(["model": model, "budget_gb": b, "error": err ?? ""]); continue }
                    _ = take(ironbark_generate(s, "Hello", 4)) // compile the kernels
                    ironbark_set_thinking(s, false)
                    let info = setImage(s, image, maxPixels: maxPixels)
                    var row: [String: Any] = ["model": model, "budget_gb": b, "image": file, "question": question,
                                              "encode": info]
                    if info["error"] == nil {
                        let messages = [["role": "user", "content": question]]
                        let start = take(ironbark_send(s, Tools.json(messages), true))
                        var text = ""
                        var n = 0
                        while start["error"] == nil && n < tokens && !self.cancel, let p = ironbark_next(s) {
                            text += String(cString: p)
                            ironbark_free_string(p)
                            n += 1
                        }
                        row["text"] = text
                        row["result"] = start["error"] == nil ? take(ironbark_stats(s)) : start
                    }
                    ironbark_close(s)
                    self.emit(row)
                }
            }
            self.append(self.cancel ? "IRONBARK_CANCELLED" : "IRONBARK_DONE")
            self.say(self.cancel ? "Stopped: the app left the screen" : "Done")
            DispatchQueue.main.async { self.busy = false }
        }
    }

    /// A scripted conversation (no tools, reasoning off) per model, budget and Expert drop:
    /// real chat text, unlike the raw prompts of `run`. One result line per turn, with the
    /// turn's stats (speed, expert cache hits, reads).
    /// `drops` empty: each model at the app's Expert drop. `minutes` > 0: the conversation
    /// again and again for that long per setting, each line with the battery level.
    func runChat(models: [String], budgets: [Double], drops: [Float], tokens: Int, minutes: Double = 0) {
        busy = true
        cancel = false
        try? FileManager.default.removeItem(at: docs.appendingPathComponent("results.jsonl"))
        let questions = ["What can I do in Adelaide?", "What else?", "Tell me about its history.", "Any food tips?"]
        UIDevice.current.isBatteryMonitoringEnabled = true // (on the main queue, as auto())
        // Read on the main queue from the bench's queue.
        let power = {
            DispatchQueue.main.sync {
                ["battery": UIDevice.current.batteryLevel, "battery_state": UIDevice.current.batteryState.rawValue,
                 "thermal": ProcessInfo.processInfo.thermalState.rawValue,
                 "low_power": ProcessInfo.processInfo.isLowPowerModeEnabled] as [String: Any]
            }
        }
        DispatchQueue.global(qos: .userInitiated).async {
            self.freeOthers()
            for model in models {
                for b in budgets {
                    for drop in drops.isEmpty ? [Float(defaultDrop(model))] : drops where !self.cancel {
                        self.say("\(model) chat, drop \(drop)…")
                        let (s, err) = open(model: model, budget: b)
                        guard let s else { self.emit(["model": model, "budget_gb": b, "error": err ?? ""]); continue }
                        _ = take(ironbark_generate(s, "Hello", 4)) // compile the kernels
                        ironbark_set_thinking(s, false)
                        ironbark_set_dynamic_k(s, drop, 3)
                        _ = take(ironbark_set_system(s, "You are a helpful assistant.", "[]"))
                        let t0 = Date()
                        self.emit(["model": model, "drop": drop, "start": power()])
                        var round = 0
                        repeat {
                            round += 1
                            var messages: [[String: Any]] = []
                            for (turn, q) in questions.enumerated() where !self.cancel {
                                messages.append(["role": "user", "content": q])
                                let start = take(ironbark_send(s, Tools.json(messages), true))
                                var text = ""
                                var n = 0
                                while start["error"] == nil && n < tokens && !self.cancel, let p = ironbark_next(s) {
                                    text += String(cString: p)
                                    ironbark_free_string(p)
                                    n += 1
                                }
                                messages.append(["role": "assistant", "content": text])
                                self.emit(["model": model, "budget_gb": b, "drop": drop, "round": round, "turn": turn + 1,
                                           "question": q, "text": String(text.prefix(300)), "tokens_out": n,
                                           "elapsed_s": Date().timeIntervalSince(t0), "power": power(),
                                           "result": take(ironbark_stats(s))])
                            }
                        } while minutes > 0 && Date().timeIntervalSince(t0) < minutes * 60 && !self.cancel
                        ironbark_close(s)
                    }
                }
            }
            self.append(self.cancel ? "IRONBARK_CANCELLED" : "IRONBARK_DONE")
            self.say(self.cancel ? "Stopped: the app left the screen" : "Done")
            DispatchQueue.main.async { self.busy = false }
        }
    }

    /// Tool-calling cases at the automatic budget (on the calling queue).
    func toolCheck(model: String, budget: Double, maxTokens: UInt32) {
        say("\(model): tool calls…")
        let (s, err) = open(model: model, budget: budget)
        guard let s else { emit(["model": model, "tools_error": err ?? ""]); return }
        _ = take(ironbark_generate(s, "Hello", 4)) // compile the kernels
        let r = take(ironbark_tool_check(s, maxTokens))
        ironbark_close(s)
        emit(["model": model, "tools": r])
        let cases = (r["results"] as? [[String: Any]] ?? []).map { ToolRow(r: $0) }
        DispatchQueue.main.async { self.tools = cases }
    }

    func runTools(model: String) {
        busy = true
        cancel = false
        tools = []
        DispatchQueue.global(qos: .userInitiated).async {
            self.freeOthers()
            self.toolCheck(model: model, budget: 4.5, maxTokens: 1024)
            self.say("Done")
            DispatchQueue.main.async { self.busy = false }
        }
    }

    func say(_ s: String) { DispatchQueue.main.async { self.log = s } }

    func auto() {
        if let turns = Int(arg("-uichat") ?? "") {
            busy = true
            try? FileManager.default.removeItem(at: docs.appendingPathComponent("results.jsonl"))
            DispatchQueue.main.async { self.busy = false } // the chat refuses to send while the bench is busy
            realChat(turns, { self.emit($0) }) { self.append("IRONBARK_DONE") }
            return
        }
        if let turns = Int(arg("-uistress") ?? "") {
            busy = true
            try? FileManager.default.removeItem(at: docs.appendingPathComponent("results.jsonl"))
            stressChat(turns, flag("-uistress-model") ? defaultModel() : nil, flag("-uistress-task"), { self.emit($0) }) {
                self.append("IRONBARK_DONE")
                DispatchQueue.main.async { self.busy = false }
            }
            return
        }
        for name in (arg("-delete") ?? "").split(separator: ",") where !name.contains("/") && name != ".." {
            try? FileManager.default.removeItem(at: docs.appendingPathComponent(String(name)))
            print("deleted \(name)")
        }
        let models = (arg("-models") ?? defaultModel()).split(separator: ",").map(String.init)
        let budgets = (arg("-budgets") ?? "0,2,1.5").split(separator: ",").compactMap { Double($0) }
        let toolTokens = UInt32(arg("-tools") ?? "0") ?? 0
        let tokens = UInt32(arg("-tokens") ?? "128") ?? 128
        let chars = Int(arg("-chars") ?? "3000") ?? 3000
        let a = fileArgs ?? CommandLine.arguments
        if flag("-chat") {
            let drops = arg("-dynamic") == "default" ? [] : (arg("-dynamic") ?? "0").split(separator: ",").compactMap { Float($0) }
            runChat(models: models, budgets: budgets, drops: drops, tokens: Int(tokens),
                    minutes: Double(arg("-minutes") ?? "0") ?? 0)
            return
        }
        if let file = arg("-image") {
            // The question may have spaces: everything after -ask up to the next option.
            var question = "Describe this photo."
            if let i = a.firstIndex(of: "-ask") {
                let words = a[(i + 1)...].prefix { !$0.hasPrefix("-") }
                if !words.isEmpty { question = words.joined(separator: " ").trimmingCharacters(in: CharacterSet(charactersIn: "\"")) }
            }
            runImage(models: models, budgets: budgets, file: file, question: question, tokens: Int(tokens),
                     maxPixels: Int(arg("-pixels") ?? "0") ?? 0)
            return
        }
        run(models: models, budgets: budgets, tokens: tokens, chars: chars, profile: a.contains("-profile"),
            toolTokens: toolTokens)
    }
}

struct BenchView: View {
    @ObservedObject var bench: Bench

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Button(bench.busy ? "Running…" : "Run (all models, device / 2 / 1.5 GB)") {
                        bench.run(models: models(), budgets: [0, 2, 1.5], tokens: 128, chars: 3000, profile: true)
                    }
                    .disabled(bench.busy)
                    Button("Tool calls (\(defaultModel()), auto)") {
                        bench.runTools(model: defaultModel())
                    }
                    .disabled(bench.busy)
                    if !bench.log.isEmpty { Text(bench.log).font(.caption).foregroundStyle(.secondary) }
                }
                if !bench.probe.isEmpty {
                    Section("Device") {
                        stat("GPU round trip", String(format: "%.0f µs", num(bench.probe, "round_trip_us")))
                        stat("Expert reads, 1 / 8 threads", String(format: "%.2f / %.2f GB/s",
                            num(bench.probe, "read", "serial_bytes_per_s") / 1e9, num(bench.probe, "read", "parallel_bytes_per_s") / 1e9))
                        stat("App memory limit", String(format: "%.2f GB", num(bench.probe, "memory", "app_limit") / 1e9))
                    }
                }
                if !bench.rows.isEmpty {
                    Section("Runs (128 tokens after a ~800-token prompt)") {
                        Grid(alignment: .trailing, horizontalSpacing: 8, verticalSpacing: 6) {
                            GridRow {
                                ForEach(["Model", "GB", "Exp.", "Prompt", "Decode", "Hit", "Wait", "Peak"], id: \.self) {
                                    Text($0).font(.caption2.bold())
                                }
                            }
                            ForEach(bench.rows) { row in
                                let r = row.r
                                GridRow {
                                    Text("\(row.model) \(row.prompt.prefix(1))").gridColumnAlignment(.leading)
                                    Text(String(format: "%.1f", row.budget == 0 ? num(r, "plan", "budget") / 1e9 : row.budget))
                                    Text(String(format: "%.0f", num(r, "experts", "per_layer")))
                                    Text(String(format: "%.0f", num(r, "prefill_tps")))
                                    Text(String(format: "%.1f", num(r, "decode_tps"))).bold()
                                    Text(String(format: "%.0f%%", 100 * num(r, "experts", "hit")))
                                    Text(String(format: "%.0fms", num(r, "experts", "wait_ms_per_tok")))
                                    Text(String(format: "%.2f", num(r, "footprint_peak") / 1e9))
                                }
                                .font(.caption2.monospacedDigit())
                            }
                        }
                        Text("Prompt and Decode in tokens/s; Exp. = experts per layer; Wait = expert loads per token; Peak = highest footprint GB during the run.")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                }
                if !bench.tools.isEmpty {
                    Section("Tool calls") {
                        ForEach(bench.tools) { t in toolCase(t.r) }
                    }
                }
                if !bench.parts.isEmpty {
                    Section("GPU time per token (no syncs, no loads)") {
                        ForEach(bench.parts) { p in
                            stat("\(p.model): \(p.name)", String(format: "%.0f µs × %.0f = %.2f ms", p.us, p.perToken, p.us * p.perToken / 1e3))
                        }
                    }
                }
            }
            .navigationTitle("Bench")
        }
    }

    /// One case: the message, the expected and actual call, speed, and the final answer.
    func toolCase(_ r: [String: Any]) -> some View {
        let pass = r["pass"] as? Bool ?? false
        let rounds = r["rounds"] as? [[String: Any]] ?? []
        let last = rounds.last ?? (r["reply"] as? [String: Any] ?? [:])
        let call = ((r["reply"] as? [String: Any])?["calls"] as? [[String: Any]])?.first
        let args = call.flatMap { try? JSONSerialization.data(withJSONObject: $0["arguments"] ?? [:]) }
            .map { String(decoding: $0, as: UTF8.self) } ?? ""
        let steps = [r["reply"] as? [String: Any] ?? [:]] + rounds
        let tokens = steps.reduce(0.0) { $0 + num($1, "tokens") }
        let secs = steps.reduce(0.0) { $0 + num($1, "seconds") }
        return VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(pass ? "✅" : "❌")
                Text(r["message"] as? String ?? "").font(.caption.bold())
            }
            Text("expect \(r["expect"] as? String ?? "no tool") · called \(r["called"] as? String ?? "none") \(args)")
                .font(.caption2.monospaced())
            Text(String(format: "%d rounds · %.0f tokens in %.1f s (%.1f tok/s)", steps.count, tokens, secs,
                        tokens / max(secs, 1e-9)))
                .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
            Text((last["text"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? "(no answer)")
                .font(.caption2).lineLimit(4)
        }
    }

    func stat(_ k: String, _ v: String) -> some View {
        HStack { Text(k); Spacer(); Text(v).monospacedDigit().foregroundStyle(.secondary) }.font(.caption)
    }
}
