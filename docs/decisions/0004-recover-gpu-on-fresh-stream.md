# DR-0004: Recover from a background GPU failure on a fresh MLX stream

| Field | Value |
|---|---|
| Status | Accepted (2026-10-25) |
| Date | 2026-10-25 |
| Deciders | project owner |
| Consulted | – |
| Related | commit bb7cb45, [DR-0002](0002-own-mlx-bindings.md) (we own the MLX bindings) |

## Table of Contents

- [Context](#context)
- [Decision drivers](#decision-drivers)
- [Options considered](#options-considered)
- [Decision matrix](#decision-matrix)
- [Decision](#decision)
- [Consequences](#consequences)
- [Follow-ups](#follow-ups)
- [References](#references)

## Context

iOS allows no GPU work in the background. Decode is pipelined: each step queues the next one before its token is read. When the app leaves the screen, the reply stops at its next token, but a step that is already queued can reach the GPU after iOS revokes access. That command buffer fails:

```
[METAL] Command buffer execution failed: Insufficient Permission (to submit GPU work
from background) (kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted)
```

MLX (0.32.4, `backend/metal/device.cpp`) stores the first failure in the stream's command encoder (`error_`) and throws it on every later operation on that stream (`get_command_encoder`, `synchronize`). There is no call to clear it. Closing the session and opening the model again does not help: the new session uses the same default GPU stream.

Seen on the iPhone 17 (bench app, `-uichat`): after a reply cut short by leaving the screen, every later message failed with this error until the app restarted.

```
 queued step ──▶ app leaves screen ──▶ command buffer fails
                                           │
                                           ▼
                          stream's encoder poisoned (error_ set)
                                           │
                     every later eval on the stream ──▶ same error
```

## Decision drivers

| # | Driver | Weight (1–3) |
|---|---|---|
| D1 | The chat works again without restarting the app | 3 |
| D2 | No change to MLX itself (we patch it only for iOS builds) | 2 |
| D3 | Small, local change | 2 |
| D4 | Nothing lost: the conversation goes on | 1 |

## Options considered

### Option A: Fresh default GPU stream, then reload

On a GPU error, make a new GPU stream the default (`mlx_set_default_stream`), open the model again (its weights load on the new stream), rebuild the conversation from the transcript and retry the message once.

- Pros:
  - Uses MLX's public C API; no patch.
  - Recovers fully: the new stream has a clean encoder.
- Cons:
  - Reloading the model and reading the conversation again takes time (prompt at ~10–40 tok/s on the phone).
  - The old stream and its encoder stay allocated (small).
- Cost / effort: S

### Option B: Patch MLX to clear the encoder's error

Add a call that resets `error_` (and poisoned events), exported through mlx-c.

- Pros:
  - No reload; the session could go on after a rebuild of its caches.
- Cons:
  - A patch to MLX internals, kept across MLX upgrades.
  - The caches written by the failed step are still wrong: the conversation must be read again anyway.
- Cost / effort: M

### Option C: Never let a step reach the GPU off screen

Stop decoding (and wait for queued work) before iOS revokes GPU access.

- Pros:
  - No failure to recover from.
- Cons:
  - iOS gives no reliable moment: the scene goes inactive, then background, and a step already committed can still be refused. Waiting on the main thread risks the watchdog.
  - Does not cover other GPU failures.
- Cost / effort: M, and not reliable

### Option D: Do nothing

The user restarts the app after leaving the screen mid-reply.

## Decision matrix

| Option | D1 (×3) | D2 (×2) | D3 (×2) | D4 (×1) | Total |
|---|---|---|---|---|---|
| A: Fresh stream, reload | 5 | 5 | 4 | 4 | 37 |
| B: Patch MLX | 4 | 1 | 2 | 4 | 22 |
| C: Never submit off screen | 2 | 5 | 2 | 5 | 25 |
| D: Do nothing | 1 | 5 | 5 | 1 | 24 |

## Decision

We choose **Option A**: on a GPU error, switch to a fresh default GPU stream and reload. It restores the chat without restarting the app and without patching MLX. The reload costs time, but the caches would need rebuilding in any option.

```
 GPU error (send fails, or a reply ends without a reason)
      │
      ▼
 broken = true ──▶ close session ──▶ ironbark_reset_gpu() ──▶ open model
      ──▶ rebuild conversation from the transcript ──▶ retry the message once
```

The app also treats a reply that ends with no finish reason as failed, because `ironbark_next` returns NULL on errors as well as at the end.

## Consequences

| Type | Item |
|---|---|
| Good | iPhone 17, reply cut by leaving the screen: 4 of 4 later messages answered (before: all failed) |
| Good | `reset_gpu` / `ironbark_reset_gpu` is in the library and the C API for other apps |
| Bad | Recovery reloads the model and reads the conversation again (tens of seconds for a long chat) |
| Bad | The old stream is kept for the life of the process |
| Risk | In the test above the failure may not have fired (timing); the recovery path itself is not yet seen succeeding → mitigation: watch for "reloading the model" in the app, add a test that forces a failed command buffer if possible |
| Risk | An MLX upgrade may change how errors are stored → mitigation: re-check `device.cpp` on upgrade |

## Follow-ups

- [ ] Confirm on device that the recovery path runs (status "The GPU was interrupted: reloading the model…") and the retried message is answered
- [ ] Revisit if MLX adds a way to clear a stream's error, or if a reply should continue in the background (continued-processing task with GPU)

## References

- MLX 0.32.4 `mlx/backend/metal/device.cpp`: `CommandEncoder::commit`, `get_command_encoder`, `synchronize`
- `crates/ironbark/src/lib.rs` (`reset_gpu`), `ios/IronbarkBench/App.swift` (`reply`, `gpuFailed`)
