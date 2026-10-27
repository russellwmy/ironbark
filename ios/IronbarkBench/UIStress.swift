// UI stress (auto.txt: -uistress TURNS; -uistress-model: a model decodes meanwhile; -uistress-task:
// inside a continued-processing task, its progress set per token as a reply does): replays tool-chat turns into the chat
// view at a reply's pace and times how long the main thread takes to answer, so a slow or
// looping view update shows as a number instead of a watchdog kill.

import BackgroundTasks
import Foundation
import UIKit

extension Chat {
    /// Replay `turns` turns (each: a message, reasoning and a tool call, its result, a long
    /// Markdown answer streamed in 0.1 s chunks); `emit` gets one result per turn.
    func stress(turns: Int, model: String?, task: Bool, emit: @escaping ([String: Any]) -> Void, done: @escaping () -> Void) {
        let result = String(repeating: #"{"extract":"Adelaide is the capital and most populous city of South Australia.","title":"Adelaide"},"#, count: 30)
        let thinking = String(repeating: "The user asks what to do in Adelaide. I should look it up first. ", count: 12)
            + "\n</think>\n\n<tool_call>\n<function=wikipedia_search>\n<parameter=query>\nAdelaide\n</parameter>\n</function>\n</tool_call>"
        let section = """
            ### Culture and museums
            - **Art Gallery of South Australia**: on North Terrace, with over 45,000 works.
            - **South Australian Museum**: natural history, with a large Aboriginal collection.
            - **Adelaide Festival Centre**: the Festival Theatre, Dunstan Playhouse and Space Theatre.

            | Place | Area | Free |
            |---|---|---|
            | Botanic Garden | North Terrace | yes |
            | Adelaide Zoo | Frome Road | no |


            """
        let answer = "Here is what you can do in Adelaide today.\n\n" + String(repeating: section, count: 6)
        // With a model: it decodes nonstop (4.5 GB) on its own queue meanwhile.
        var running = true
        let decoding = DispatchSemaphore(value: 0)
        if let model {
            DispatchQueue.global(qos: .userInitiated).async {
                let (s, err) = open(model: model, budget: 4.5)
                if let s {
                    while running { _ = take(ironbark_generate(s, "Tell me about Adelaide in detail.", 64)) }
                    ironbark_close(s)
                } else {
                    emit(["uistress_model_error": err ?? ""])
                }
                decoding.signal()
            }
        } else {
            decoding.signal()
        }
        let replay: (AnyObject?) -> Void = { bgTask in
            if #available(iOS 26, *), let t = bgTask as? BGContinuedProcessingTask { t.progress.totalUnitCount = 1_000_000 }
            emit(["uistress_in_task": bgTask != nil])
            for turn in 0..<turns {
                // Main-queue latency, sampled every 50 ms while the turn plays.
                var worst = 0.0, slow = 0, playing = true
                let probe = Thread {
                    while playing {
                        let t = Date()
                        let answered = DispatchSemaphore(value: 0)
                        DispatchQueue.main.async { answered.signal() }
                        _ = answered.wait(timeout: .now() + 30)
                        let ms = Date().timeIntervalSince(t) * 1000
                        worst = max(worst, ms)
                        if ms > 250 { slow += 1 }
                        Thread.sleep(forTimeInterval: 0.05)
                    }
                }
                probe.start()
                let start = Date()
                func main(_ f: @escaping () -> Void) { DispatchQueue.main.async(execute: f) }
                func stream(_ text: String) {
                    main { self.messages.append(Message(role: .assistant)) }
                    var rest = Substring(text)
                    while !rest.isEmpty {
                        let piece = String(rest.prefix(60))
                        rest = rest.dropFirst(60)
                        main { self.messages[self.messages.count - 1].text += piece }
                        // A token every ~40 ms, each setting the task's progress (as `reply` does).
                        for _ in 0..<3 {
                            if #available(iOS 26, *), let t = bgTask as? BGContinuedProcessingTask {
                                t.progress.completedUnitCount += 1
                            }
                            Thread.sleep(forTimeInterval: 0.033)
                        }
                    }
                }
                main { self.messages.append(Message(role: .user, text: "What else can I do? (\(turn + 1))")) }
                stream(thinking)
                main {
                    var tool = Message(role: .tool)
                    tool.calls = [("wikipedia_search(query: Adelaide)", "[\(result)]")]
                    self.messages.append(tool)
                }
                stream("Found it.\n</think>\n\n" + answer)
                main { self.status = "turn \(turn + 1) · \(self.messages.count) messages" }
                playing = false
                while !probe.isFinished { Thread.sleep(forTimeInterval: 0.01) }
                let row: [String: Any] = ["uistress_turn": turn + 1, "messages": (turn + 1) * 4,
                                          "seconds": Date().timeIntervalSince(start),
                                          "worst_main_ms": worst, "stalls_over_250ms": slow]
                emit(row)
            }
            running = false
            decoding.wait()
            if #available(iOS 26, *), let t = bgTask as? BGContinuedProcessingTask { t.setTaskCompleted(success: true) }
            done()
        }
        if !(task && continueInBackground(replay)) {
            DispatchQueue.global(qos: .userInitiated).async { replay(nil) }
        }
    }
}

extension Chat {
    /// The real chat (auto.txt: -uichat TURNS): send questions through `send` (model, tools,
    /// Markdown), one after another, timing the main thread as `stress` does.
    func stressReal(turns: Int, emit: @escaping ([String: Any]) -> Void, done: @escaping () -> Void) {
        let questions = ["What can I do in Adelaide?", "What else", "What is the weather there?",
                         "Tell me about its history", "What else", "Any food tips?"]
        DispatchQueue.global(qos: .userInitiated).async {
            for turn in 0..<turns {
                var worst = 0.0, slow = 0, playing = true
                let probe = Thread {
                    while playing {
                        let t = Date()
                        let answered = DispatchSemaphore(value: 0)
                        DispatchQueue.main.async { answered.signal() }
                        _ = answered.wait(timeout: .now() + 30)
                        let ms = Date().timeIntervalSince(t) * 1000
                        worst = max(worst, ms)
                        if ms > 250 { slow += 1 }
                        Thread.sleep(forTimeInterval: 0.05)
                    }
                }
                probe.start()
                // As a user: send only with the app on screen.
                while DispatchQueue.main.sync(execute: { UIApplication.shared.applicationState != .active }) {
                    Thread.sleep(forTimeInterval: 0.2)
                }
                let start = Date()
                DispatchQueue.main.sync { _ = self.send(questions[turn % questions.count]) }
                while DispatchQueue.main.sync(execute: { self.busy }) { Thread.sleep(forTimeInterval: 0.2) }
                playing = false
                while !probe.isFinished { Thread.sleep(forTimeInterval: 0.01) }
                let (count, chars, status) = DispatchQueue.main.sync {
                    (self.messages.count, self.messages.reduce(0) { $0 + $1.text.count }, self.status)
                }
                emit(["uichat_turn": turn + 1, "messages": count, "chars": chars, "seconds": Date().timeIntervalSince(start),
                      "worst_main_ms": worst, "stalls_over_250ms": slow, "status": status])
            }
            done()
        }
    }
}

/// Always on: when the main thread does not answer for 2 s, write what the chat was doing to
/// Documents/stalls.jsonl (the watchdog kills the app at 10 s, so this is what is left).
enum StallLog {
    private static let lock = NSLock()
    private static var state: [String: Any] = [:]

    /// Note the chat's state (main thread, on each change).
    static func note(_ fields: [String: Any]) {
        lock.lock()
        state.merge(fields) { $1 }
        lock.unlock()
    }

    static func start() {
        Thread.detachNewThread {
            Thread.current.name = "stall-log"
            while true {
                Thread.sleep(forTimeInterval: 0.5)
                let t = Date()
                let answered = DispatchSemaphore(value: 0)
                DispatchQueue.main.async { answered.signal() }
                if answered.wait(timeout: .now() + 2) == .success { continue }
                lock.lock()
                var row = state
                lock.unlock()
                row["stall_at"] = t.formatted(.iso8601)
                write(row)
                // Wait it out, then note how long it lasted.
                answered.wait()
                write(["stall_at": t.formatted(.iso8601), "stall_s": Date().timeIntervalSince(t)])
            }
        }
    }

    private static func write(_ row: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: row) else { return }
        let url = docs.appendingPathComponent("stalls.jsonl")
        let line = data + Data("\n".utf8)
        if let h = try? FileHandle(forWritingTo: url) {
            h.seekToEndOfFile()
            h.write(line)
            try? h.close()
        } else {
            try? line.write(to: url)
        }
    }
}
