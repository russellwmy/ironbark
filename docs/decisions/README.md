# Decision Records (DR)

A Decision Record captures one important decision: what we chose, why, and what we gave up.

## Table of Contents

- [When to write a DR](#when-to-write-a-dr)
- [Lifecycle](#lifecycle)
- [How to write one](#how-to-write-one)
- [Naming and numbering](#naming-and-numbering)
- [Rules](#rules)
- [Index](#index)

## When to write a DR

Write one when the decision is:

| Signal | Example |
|---|---|
| Hard to reverse | Database choice, wire protocol, public API shape |
| Affects many parts | Error handling style, auth model, crate layout |
| Has real trade-offs | Speed vs. safety, build-vs-buy |
| Will be questioned later | "Why didn't we just use X?" |

Skip it for small, local, easy-to-undo choices.

## Lifecycle

```
              +-----------+
   write ---> | Proposed  |
              +-----+-----+
                    |
         review     |
        +-----------+-----------+
        v                       v
  +-----------+           +-----------+
  | Accepted  |           | Rejected  |
  +-----+-----+           +-----------+
        |
        | later, a new DR changes it
        v
  +-----------+     or    +-------------+
  | Superseded|           | Deprecated  |
  | by DR-NNNN|           | (no longer  |
  +-----------+           |  relevant)  |
                          +-------------+
```

| Status | Meaning |
|---|---|
| Proposed | Draft, open for review |
| Accepted | Agreed, in effect |
| Rejected | Considered and declined (keep it, it's still useful) |
| Superseded | Replaced by a newer DR (link to it) |
| Deprecated | No longer applies, nothing replaced it |

## How to write one

1. Copy `0000-template.md` to `NNNN-short-title.md`.
2. Fill in **Context** and **Options** first. Decide last.
3. Score options against the drivers in the decision matrix.
4. Open a PR with status `Proposed`. Discuss in the PR.
5. On merge, set status to `Accepted` and add a row to the [Index](#index).

```
 Problem --> Drivers --> Options --> Matrix --> Decision --> Consequences
   why        what        2-4        score      pick one     good / bad /
  now?       matters     choices                + reason     follow-ups
```

## Naming and numbering

- File: `NNNN-kebab-case-title.md` (e.g. `0001-use-postgres-for-state.md`)
- Numbers are sequential and never reused.
- Title states the decision, not the question: "Use Postgres for state", not "Which database?"

## Rules

- One decision per DR.
- Never edit an Accepted DR's decision. Write a new DR that supersedes it, and update the old one's status line only.
- Keep it short. 1–2 pages is the target.
- Always list at least two options, including "do nothing" when it's real.
- Write down the downsides of the chosen option.

## Index

| # | Title | Status | Date |
|---|---|---|---|
| [0000](0000-template.md) | Template | — | — |
| [0001](0001-ternary-moe-streaming.md) | Save RAM with a ternary MoE streamed by ironbark | Accepted | 2026-10-14 |
| [0002](0002-own-mlx-bindings.md) | Own the MLX bindings in house (ironbark-mlx) | Accepted | 2026-10-16 |
