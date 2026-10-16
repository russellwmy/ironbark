// IronbarkBench: chat with Maple and benchmark it on iPhone, through the C API in ironbark.h. Models
// are `ironbark pack` dirs in Documents (default "maple"). The chat can call the tools in
// Tools.swift and renders replies with Markdown.swift.
//
// Benchmark without the UI: put arguments in Documents/auto.txt (or pass them at launch with
// -auto); the app runs them at its next launch, deletes the file, and writes one JSON line per
// result to Documents/results.jsonl, ending with IRONBARK_DONE (IRONBARK_CANCELLED if the app
// left the screen).
//   -models maple,maple-2bit  -budgets 0,2,1.5 (GB; 0 = from the device)  -tokens 128
//   -chars 3000  -profile  -prompts prose,rust
//   -lookahead 1,2 (each budget at each lookahead; default 1)  -flash (each also with FlashHead)
//   -tools 1024 (tool-calling cases after the runs, replies of at most 1024 tokens; -budgets ""
//   skips the runs)

import BackgroundTasks
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
                // One model in memory at a time: the bench frees the chat's, and the chat waits.
                bench.freeOthers = chat.free
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

    /// As ironbark_send reads it.
    var json: [String: Any] {
        ["role": role.rawValue, "content": role == .tool ? calls.map(\.result) as Any : text]
    }
}

/// A streamed reply split for display: reasoning, answer, and whether it calls a tool. The
/// prompt opens the reasoning, so a reply starts inside it (as ironbark::parse).
struct Shown {
    var thinking = "", answer = "", calling = false, thought = false

    init(_ raw: String) {
        var rest = Substring(raw)
        if let end = raw.range(of: "</think>") {
            thinking = String(raw[..<end.lowerBound])
            rest = raw[end.upperBound...]
            thought = true
        } else if !raw.contains("<tool_call>") {
            thinking = raw
            rest = ""
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
    @Published var busy = false
    @Published var status = ""
    @Published var model = "maple"
    @Published var budget = 0.0
    /// Reason before answering a message (replies to tool results always do).
    @Published var think = false

    /// Every library call runs here, one at a time.
    private let engine = DispatchQueue(label: "ironbark.engine", qos: .userInitiated)
    // Engine queue only:
    private var session: OpaquePointer?
    private var opened: (String, Double)?
    /// Messages of the transcript the session's KV cache holds; otherwise the next turn
    /// rebuilds the conversation from the transcript.
    private var cached = 0
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
    func send(_ question: String) -> Bool {
        if otherBusy() {
            status = "The bench is running: send after it finishes"
            return false
        }
        let history = messages
        messages.append(Message(role: .user, text: question))
        let (model, budget, think) = (self.model, self.budget, self.think)
        busy = true
        stop = false
        let job: (AnyObject?) -> Void = { task in self.reply(question, history, model, budget, think, task) }
        if !continueInBackground(job) {
            engine.async { job(nil) }
        }
        return true
    }

    /// iOS 26: run the reply as a continued-processing task, so it can finish on the GPU if the
    /// app leaves the screen. False when the device or the app is not allowed to.
    private func continueInBackground(_ job: @escaping (AnyObject?) -> Void) -> Bool {
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
        _ question: String, _ history: [Message], _ model: String, _ budget: Double, _ think: Bool, _ task: AnyObject?
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
                let system = "You are a helpful assistant. Today is \(date). Answer in Markdown when it helps."
                _ = take(ironbark_set_system(s, system, Tools.schemasJSON))
            }
        }
        guard let s = session else { return }
        ironbark_set_thinking(s, think)
        // Continue the conversation in the KV cache, or rebuild it from the transcript.
        let rebuild = cached != history.count
        var send = (rebuild ? history : []).map(\.json) + [Message(role: .user, text: question).json]
        var held = history.count + 1
        var count = 0
        for round in 0...Self.maxRounds {
            let start = take(ironbark_send(s, Tools.json(send), rebuild && round == 0))
            if let e = start["error"] { cached = 0; finish("Error: \(e)", task); return }
            DispatchQueue.main.async { self.messages.append(Message(role: .assistant)) }
            var raw = ""
            var ended = false
            var tokens = 0
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
                DispatchQueue.main.async { self.messages[self.messages.count - 1].text += piece }
                if tokens >= Self.maxTokens { break }
            }
            held += 1
            // The session closes a reply cut short, so the KV cache still matches the transcript.
            let calls = ended ? take(ironbark_parse(raw))["calls"] as? [[String: Any]] ?? [] : []
            if calls.isEmpty || round == Self.maxRounds || stop { break }
            setStatus("Running \(calls.compactMap { $0["name"] as? String }.joined(separator: ", "))…")
            var tool = Message(role: .tool)
            for c in calls {
                let name = c["name"] as? String ?? ""
                let args = c["arguments"] as? [String: Any] ?? [:]
                let shown = args.sorted { $0.key < $1.key }.map { "\($0.key): \($0.value)" }.joined(separator: ", ")
                tool.calls.append(("\(name)(\(shown))", Tools.run(name, args)))
            }
            DispatchQueue.main.async { self.messages.append(tool) }
            held += 1
            send = [tool.json]
        }
        cached = held
        let st = take(ironbark_stats(s))
        let line = String(
            format: "%.1f tok/s · prompt %.0f tok/s · %.0f experts/layer · hit %.0f%% · footprint %.2f GB (peak %.2f)",
            num(st, "decode_tps"), num(st, "prefill_tps"), num(st, "experts", "per_layer"),
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
        messages = []
        status = ""
        engine.async { self.cached = 0 }
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

struct MessageView: View {
    let m: Message

    var body: some View {
        switch m.role {
        case .user:
            Text(m.text)
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

struct ChatView: View {
    @ObservedObject var chat: Chat
    @State private var input = ""
    @FocusState private var typing: Bool

    var body: some View {
        NavigationStack {
            VStack(spacing: 8) {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 10) {
                            ForEach(chat.messages) { m in
                                MessageView(m: m)
                                    .frame(maxWidth: .infinity, alignment: m.role == .user ? .trailing : .leading)
                                    .id(m.id)
                            }
                        }
                        .padding(.horizontal)
                    }
                    // Scrolling or tapping the messages hides the keyboard.
                    .scrollDismissesKeyboard(.interactively)
                    .onTapGesture { typing = false }
                    .onChange(of: chat.messages.count) { _, _ in
                        if let id = chat.messages.last?.id { proxy.scrollTo(id, anchor: .bottom) }
                    }
                    .onChange(of: chat.messages.last?.text) { _, _ in
                        if let id = chat.messages.last?.id { proxy.scrollTo(id, anchor: .bottom) }
                    }
                }
                if !chat.status.isEmpty {
                    Text(chat.status).font(.caption2).foregroundStyle(.secondary).padding(.horizontal)
                }
                HStack {
                    TextField("Message", text: $input, axis: .vertical).textFieldStyle(.roundedBorder)
                        .focused($typing)
                    if chat.busy {
                        Button("Stop") { chat.cancel() }
                    } else {
                        Button("Send") {
                            let q = input.trimmingCharacters(in: .whitespacesAndNewlines)
                            guard !q.isEmpty, chat.send(q) else { return }
                            input = ""
                            typing = false
                        }
                    }
                }
                .padding([.horizontal, .bottom])
            }
            .navigationTitle("Maple")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Menu {
                        Picker("Model", selection: $chat.model) { ForEach(models(), id: \.self) { Text($0) } }
                        Toggle("Think", isOn: $chat.think)
                        Picker("Budget", selection: $chat.budget) {
                            ForEach([0.0, 1.5, 2.0, 3.0], id: \.self) { Text($0 == 0 ? "Auto" : String(format: "%.1f GB", $0)) }
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
                    for la in lookaheads {
                        ironbark_set_lookahead(s, la)
                        for flash in flashes {
                            _ = take(ironbark_set_flash_head(s, flash))
                            for name in prompts where !self.cancel {
                                let r = take(ironbark_generate(s, self.prompt(name, chars: chars), tokens))
                                self.emit(["model": model, "budget_gb": b, "prompt": name, "lookahead": la, "flash": flash,
                                           "result": r])
                                DispatchQueue.main.async {
                                    self.rows.append(Row(model: model, budget: b, prompt: name, r: r))
                                }
                            }
                        }
                    }
                    ironbark_close(s)
                }
            }
            if toolTokens > 0 && !self.cancel, let model = models.first {
                self.toolCheck(model: model, maxTokens: toolTokens)
            }
            self.append(self.cancel ? "IRONBARK_CANCELLED" : "IRONBARK_DONE")
            self.say(self.cancel ? "Stopped: the app left the screen" : "Done")
            DispatchQueue.main.async { self.busy = false }
        }
    }

    /// Tool-calling cases at the automatic budget (on the calling queue).
    func toolCheck(model: String, maxTokens: UInt32) {
        say("\(model): tool calls…")
        let (s, err) = open(model: model, budget: 0)
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
            self.toolCheck(model: model, maxTokens: 1024)
            self.say("Done")
            DispatchQueue.main.async { self.busy = false }
        }
    }

    func say(_ s: String) { DispatchQueue.main.async { self.log = s } }

    func auto() {
        let models = (arg("-models") ?? "maple").split(separator: ",").map(String.init)
        let budgets = (arg("-budgets") ?? "0,2,1.5").split(separator: ",").compactMap { Double($0) }
        let toolTokens = UInt32(arg("-tools") ?? "0") ?? 0
        let tokens = UInt32(arg("-tokens") ?? "128") ?? 128
        let chars = Int(arg("-chars") ?? "3000") ?? 3000
        let a = fileArgs ?? CommandLine.arguments
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
                    Button("Tool calls (\(models().first ?? "maple"), auto)") {
                        bench.runTools(model: models().first ?? "maple")
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
