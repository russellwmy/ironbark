/* C API of ironbark-ffi: run Maple (ternary MoE) streamed from a packed dir (`ironbark pack`).
 * Functions that return char* return JSON; free it with ironbark_free_string.
 * On failure the JSON is {"error": "..."}. */
#ifndef IRONBARK_H
#define IRONBARK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Session IronbarkSession;

/* Open the packed model in `dir` sized to a footprint of `budget` bytes (0: from the device,
 * 3/8 of RAM and at most half the iOS app limit). `slots` overrides the expert cache
 * slots per layer (0: from the budget; 256 keeps every expert loaded).
 * Null on failure; *error gets a message if not null. */
IronbarkSession *ironbark_open(const char *dir, uint64_t budget, uint32_t slots, char **error);

/* Fetcher: builds a local experts.bin (or embed.bin) from a remote copy, apart from any session. */
typedef struct Fetcher IronbarkFetcher;

/* Fill `buf` with `len` bytes of the remote file (experts.bin, or embed.bin for an embed
 * fetcher) from `offset` (e.g. an HTTP Range GET); return 0 on success. Called from several
 * threads at once. */
typedef int32_t (*IronbarkFetch)(void *ctx, uint64_t offset, uint8_t *buf, size_t len);

/* Open the fetcher of `dir`, whose experts.bin may be missing or partial (experts.json and
 * manifest.json present). Records come through `fetch`, are checked against manifest.json
 * and kept on disk. `fetch` and `ctx` must stay valid until the fetcher and every session
 * using it are closed. Null on failure; *error gets a message if not null. */
IronbarkFetcher *ironbark_fetcher_open(const char *dir, IronbarkFetch fetch, void *ctx, char **error);
/* The same for embed.bin (manifest.json present; `fetch` reads the remote embed.bin, checked
 * per chunk). A session fetches the rows each token needs; start background threads once the
 * experts are in (they compete for the network) to complete it. */
IronbarkFetcher *ironbark_embed_fetcher_open(const char *dir, IronbarkFetch fetch, void *ctx, char **error);
/* Fetch every missing record on `threads` background threads (they hold off while a session
 * waits on a fetch); stop waits for them. A session still fetches what it needs. */
void ironbark_fetcher_start(const IronbarkFetcher *f, uint32_t threads);
void ironbark_fetcher_stop(const IronbarkFetcher *f);
/* JSON: present, total, complete, fetched_bytes, errors, last_error. Any thread. */
char *ironbark_fetcher_progress(const IronbarkFetcher *f);
/* Stop and release; sessions using it keep working. */
void ironbark_fetcher_close(IronbarkFetcher *f);

/* Like ironbark_open, reading expert records through `fetcher`: one not on disk yet is fetched
 * when the router first needs it. */
IronbarkSession *ironbark_open_fetcher(const char *dir, uint64_t budget, uint32_t slots,
                               const IronbarkFetcher *fetcher, char **error);
/* Like ironbark_open, with experts.bin read through `experts` (ironbark_fetcher_open) and
 * embed.bin through `embed` (ironbark_embed_fetcher_open). Either may be null: that file must
 * then be complete. */
IronbarkSession *ironbark_open_fetchers(const char *dir, uint64_t budget, uint32_t slots,
                                const IronbarkFetcher *experts, const IronbarkFetcher *embed,
                                char **error);

/* Greedy-generate `tokens` tokens after `prompt`: speed, text, expert stats, memory. */
char *ironbark_generate(IronbarkSession *s, const char *prompt, uint32_t tokens);

/* GPU round trip, expert reads from dir/experts.bin (dir may be null), memory. */
char *ironbark_probe(const char *dir);

/* Chat: read `text` (a formatted turn) into the conversation; reset = start a new one.
 * Then call ironbark_next until it returns NULL (end of the reply); each call returns the next
 * piece of text (possibly ""). ironbark_stats reports the turn's speed and memory. */
char *ironbark_start(IronbarkSession *s, const char *text, bool reset);
char *ironbark_next(IronbarkSession *s);
char *ironbark_stats(IronbarkSession *s);

/* Chat with tools: start every conversation with a system turn of `system` (may be NULL) and
 * the tools in `tools` (a JSON array of function schemas, may be NULL); forgets the
 * conversation. Returns {}. */
char *ironbark_set_system(IronbarkSession *s, const char *system, const char *tools);
/* Read `messages` (JSON array of {"role": "user"|"assistant", "content": text} and
 * {"role": "tool", "content": [result, ...]}, ending in a user or tool message) into the
 * conversation, `reset` first forgets it (to rebuild a transcript); then stream the reply
 * with ironbark_next. Returns the turn's stats. */
char *ironbark_send(IronbarkSession *s, const char *messages, bool reset);
/* Whether replies reason first (default on). Off, a reply to a user message answers at once
 * and starts with "</think>"; replies to tool results still reason. The system turn is read
 * once per session and restored for each new conversation. */
void ironbark_set_thinking(IronbarkSession *s, bool on);
/* Streaming: layers ahead whose experts each step predicts and prefetches (default 1). */
void ironbark_set_lookahead(IronbarkSession *s, uint32_t layers);
/* Decode with FlashHead, DeepGrove's approximate LM head (exact logits only for the top 512
 * of 4748 vocabulary clusters), or the exact head (default). Returns {"available": bool}. */
char *ironbark_set_flash_head(IronbarkSession *s, bool on);
/* Split a reply: JSON thinking, text, calls [{name, arguments}], malformed. */
char *ironbark_parse(const char *reply);

/* GPU time of each part of a decode step (no syncs, no expert loads). */
char *ironbark_profile(IronbarkSession *s);

/* Tool-calling cases (web search, weather, calculator, no tool), replies of at most
 * max_tokens each. Forgets the conversation. */
char *ironbark_tool_check(IronbarkSession *s, uint32_t max_tokens);

void ironbark_close(IronbarkSession *s);
void ironbark_free_string(char *p);

#ifdef __cplusplus
}
#endif

#endif
