# SGLang Production Bugfixes

This directory tracks bugs discovered in **production environments**, along with the fixes we applied locally or on forked branches.

Upstream fixes are submitted to the SGLang community when possible. However, community review and merge timelines are unpredictable — patches may be delayed, rejected, or superseded. To avoid losing context and to unblock production deployments, we document each issue here: **symptoms**, **root cause**, **branch/commit**, and **related changes**.

---

## How to Add an Entry

Copy the template below into a new section (or a dedicated markdown file under this directory) for each bugfix.

---

## Bugfix

### [Fix inconsistent data types in detokenizer](https://github.com/shinemo-ai/sglang/tree/fix-detokenize-tensor-bug)

| Field | Details |
|-------|---------|
| **Status** | `None` |
| **Severity** |  `medium` |
| **SGLang version** | `v0.5.12` |
| **Branch** | `fix-detokenize-tensor-bug` |
| **Commits** | `323e8cd` |
| **Upstream PR** | `N/A` |

#### Symptoms

If the request contains `logprob` related parameters, it may trigger and cause the service to crash.

```bash
TokenizerManager hit an exception: Traceback (most recent call last):
File "/sgl-workspace/sglang/python/sglang/srt/managers/tokenizer_manager.py", line 2749, in print_exception_wrapper
File "/sgl-workspace/sglang/python/sglang/srt/managers/tokenizer_manager.py", line 1654, in handle_loop
File "/sgl-workspace/sglang/python/sglang/srt/managers/tokenizer_manager.py", line 1696, in _handle_batch_output
File "/sgl-workspace/sglang/python/sglang/srt/managers/tokenizer_manager.py", line 2055, in convert_logprob_style
File "/sgl-workspace/sglang/python/sglang/srt/managers/tokenizer_manager.py", line 2093, in detokenize_top_logprobs_tokens
File "/sgl-workspace/sglang/python/sglang/srt/managers/tokenizer_manager.py", line 1951, in add_logprob_to_meta_info
RuntimeError: Boolean value of Tensor with more than one value is ambiguous
```