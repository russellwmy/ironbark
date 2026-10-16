// Markdown for replies. SwiftUI's Text renders only inline Markdown (bold, italic, code, links),
// so the blocks are split here: headings, lists, quotes, fenced code, tables and rules, each
// with inline styling. A reply still streaming (an open code fence) renders as it stands.

import SwiftUI

enum Block {
    case heading(Int, String)
    case paragraph(String)
    /// Indent level, marker ("•" or "3."), text.
    case item(Int, String, String)
    case quote(String)
    case code(String)
    case table([[String]])
    case rule
}

func blocks(_ text: String) -> [Block] {
    var out: [Block] = []
    var para: [String] = []
    var quote: [String] = []
    var table: [[String]] = []
    var code: [String]?
    func flush() {
        if !para.isEmpty { out.append(.paragraph(para.joined(separator: "\n"))); para = [] }
        if !quote.isEmpty { out.append(.quote(quote.joined(separator: "\n"))); quote = [] }
        if !table.isEmpty { out.append(.table(table)); table = [] }
    }
    for raw in text.components(separatedBy: "\n") {
        let line = raw.trimmingCharacters(in: .whitespaces)
        if let c = code {
            if line.hasPrefix("```") {
                out.append(.code(c.joined(separator: "\n")))
                code = nil
            } else {
                code = c + [raw]
            }
            continue
        }
        if line.hasPrefix("```") { flush(); code = []; continue }
        if line.isEmpty { flush(); continue }
        if line.hasPrefix("|") {
            if table.isEmpty { flush() }
            let cells = line.trimmingCharacters(in: CharacterSet(charactersIn: "|")).components(separatedBy: "|")
                .map { $0.trimmingCharacters(in: .whitespaces) }
            // The |---|:--:| row under the header.
            if !cells.allSatisfy({ !$0.isEmpty && $0.allSatisfy { "-:".contains($0) } }) { table.append(cells) }
            continue
        }
        if !table.isEmpty { flush() }
        if line.allSatisfy({ $0 == "-" || $0 == " " }) && line.filter({ $0 == "-" }).count >= 3
            || line.allSatisfy({ $0 == "*" || $0 == " " }) && line.filter({ $0 == "*" }).count >= 3
            || line.allSatisfy({ $0 == "_" }) && line.count >= 3 {
            flush(); out.append(.rule); continue
        }
        if let h = line.firstIndex(where: { $0 != "#" }), line.hasPrefix("#"), line[h] == " ",
           line.distance(from: line.startIndex, to: h) <= 6 {
            flush()
            out.append(.heading(line.distance(from: line.startIndex, to: h), String(line[h...]).trimmingCharacters(in: .whitespaces)))
            continue
        }
        if line.hasPrefix(">") {
            if !para.isEmpty { flush() }
            quote.append(String(line.dropFirst()).trimmingCharacters(in: .whitespaces))
            continue
        }
        let indent = raw.prefix { $0 == " " || $0 == "\t" }.reduce(0) { $0 + ($1 == "\t" ? 4 : 1) } / 2
        if let first = line.first, "-*+".contains(first), line.dropFirst().first == " " {
            flush(); out.append(.item(indent, "•", String(line.dropFirst(2)))); continue
        }
        if let dot = line.firstIndex(where: { !$0.isNumber }), dot > line.startIndex, ".)".contains(line[dot]),
           line[line.index(after: dot)...].first == " " {
            flush()
            out.append(.item(indent, String(line[...dot]), String(line[line.index(dot, offsetBy: 2)...])))
            continue
        }
        if !quote.isEmpty { flush() }
        para.append(line)
    }
    if let c = code { out.append(.code(c.joined(separator: "\n"))) }
    flush()
    return out
}

/// Inline Markdown (bold, italic, code, links, strikethrough), keeping line breaks.
func inline(_ s: String) -> AttributedString {
    (try? AttributedString(markdown: s, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)))
        ?? AttributedString(s)
}

struct MarkdownView: View {
    let text: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(blocks(text).enumerated()), id: \.offset) { _, b in block(b) }
        }
    }

    @ViewBuilder private func block(_ b: Block) -> some View {
        switch b {
        case let .heading(level, s):
            Text(inline(s)).font(level == 1 ? .title3.bold() : level == 2 ? .headline : .subheadline.bold())
                .padding(.top, 4)
        case let .paragraph(s):
            Text(inline(s))
        case let .item(indent, marker, s):
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text(marker).monospacedDigit()
                Text(inline(s))
            }
            .padding(.leading, CGFloat(indent) * 16)
        case let .quote(s):
            Text(inline(s)).foregroundStyle(.secondary)
                .padding(.leading, 10)
                .overlay(alignment: .leading) { Rectangle().fill(.secondary.opacity(0.5)).frame(width: 3) }
        case let .code(s):
            ScrollView(.horizontal, showsIndicators: false) {
                Text(s).font(.caption.monospaced()).padding(8)
            }
            .background(Color.secondary.opacity(0.12))
            .clipShape(RoundedRectangle(cornerRadius: 6))
        case let .table(rows):
            ScrollView(.horizontal, showsIndicators: false) {
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
                    ForEach(Array(rows.enumerated()), id: \.offset) { r, cells in
                        GridRow {
                            ForEach(Array(cells.enumerated()), id: \.offset) { _, c in
                                Text(inline(c)).fontWeight(r == 0 ? .semibold : .regular)
                            }
                        }
                        if r == 0 { Divider() }
                    }
                }
                .font(.callout)
            }
        case .rule:
            Divider()
        }
    }
}
