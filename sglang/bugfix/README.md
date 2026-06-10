# SGLang Production Bugfixes

This directory tracks bugs discovered in **production environments**, along with the fixes we applied locally or on forked branches.

Upstream fixes are submitted to the SGLang community when possible. However, community review and merge timelines are unpredictable — patches may be delayed, rejected, or superseded. To avoid losing context and to unblock production deployments, we document each issue here: **symptoms**, **root cause**, **branch/commit**, and **related changes**.

---

## How to Add an Entry

Copy the template below into a new section for each bugfix.

```markdown
## [Short title](link-to-branch-or-pr)

| Field | Details |
|-------|---------|
| **Status** | `fixed` / `pending upstream` / `workaround` |
| **Severity** | `low` / `medium` / `high` |
| **SGLang version** | `vX.Y.Z` |
| **Branch** | `branch-name` |
| **Commits** | `abc1234` |
| **Upstream PR** | `N/A` or PR link |

#### Symptoms

Describe what users or operators observe in production.

#### Root Cause

Explain why the bug happens.

#### Related Changes

Link to the fix and summarize the key code change.
```

---

## [Fix inconsistent data types in detokenizer](https://github.com/shinemo-ai/sglang/tree/fix-detokenize-tensor-bug)

| Field | Details |
|-------|---------|
| **Status** | `None` |
| **Severity** | `medium` |
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

#### Related Changes

Fixed in [detokenize_top_logprobs_tokens](https://github.com/shinemo-ai/sglang/blob/fix-detokenize-tensor-bug/python/sglang/srt/managers/tokenizer_manager.py#L2134):

```python
def detokenize_top_logprobs_tokens(
    self,
    token_logprobs_val: List[float],
    token_logprobs_idx: List[int],
    decode_to_text: bool,
):
    # TODO: The current implementation only batches the detokenization for top-k tokens per single position.
    # We should batch all top-k tokens in all positions.
    ret = []
    for i in range(len(token_logprobs_val)):
        val, idx = token_logprobs_val[i], token_logprobs_idx[i]
        if isinstance(val, torch.Tensor):
            if val.numel() == 0:
                ret.append(None)
                continue
            val = val.tolist()
        elif not val:
            ret.append(None)
            continue
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        ret.append(self.detokenize_logprob_tokens(val, idx, decode_to_text))
    return ret
```

---

## [Fix prefill-decode disaggregation router bug when returning logprobs](https://github.com/shinemo-ai/sglang/tree/fix-pd-router-bug)

| Field | Details |
|-------|---------|
| **Status** | `None` |
| **Severity** | `high` |
| **Sgl-model-gateway version** | `v0.3.1` |
| **Branch** | `fix-pd-router-bug` |
| **Commits** | `a3d72cb` |
| **Upstream PR** | `N/A` |

#### Symptoms

In PD disaggregation, if `sgl-maas-gateway` is used, an error will occur when a user requests and passes in `logprob` related parameters. The main reason is that in Rust, the `bytes` type returns `application/octet-stream` by default, causing JSON parsing to fail.

#### Related Changes

Fixed in [pd_router](https://github.com/shinemo-ai/sglang/commit/a3d72cb65f0b4685c1f13b9cb46e3bfe78c997c4).

---

## [Fix AttributeError in encode_arguments_to_dsml for DeepSeek-V4](https://github.com/sgl-project/sglang/pull/25658)

This bug fix comes from contributions from the open-source community.

| Field | Details |
|-------|---------|
| **Status** | `None` |
| **Severity** | `medium` |
| **SGLang version** | `v0.5.12` |
| **Branch** | `None` |
| **Commits** | `None` |
| **Upstream PR** | `N/A` |

#### Related Changes

Fixed in [encode_arguments_to_dsml](https://github.com/sgl-project/sglang/pull/25658):

```python
def encode_arguments_to_dsml(tool_call: Dict[str, str]) -> str:
    """
    Encode tool call arguments into DSML parameter format.

    Args:
        tool_call: Dict with "name" and "arguments" (JSON string) keys.

    Returns:
        DSML-formatted parameter string.
    """
    p_dsml_template = '<{dsml_token}parameter name="{key}" string="{is_str}">{value}</{dsml_token}parameter>'
    P_dsml_strs = []

    try:
        arguments = json.loads(tool_call["arguments"])
    except Exception as err:
        arguments = tool_call["arguments"]

    if not isinstance(arguments, dict):
        arguments = {
            "arguments": arguments if isinstance(arguments, str) else to_json(arguments)
        }

    for k, v in arguments.items():
        ...
```

---

## [Fix a bug in xgrammar in thinking mode](https://github.com/shinemo-ai/sglang/commit/0590737450afd8ef0577ed1a7c232cb116208a70)

| Field | Details |
|-------|---------|
| **Status** | `None` |
| **Severity** | `high` |
| **SGLang version** | `v0.5.12` |
| **Branch** | `fix-reason-grammar` |
| **Commits** | `0590737` |
| **Upstream PR** | `N/A` |

#### Symptoms

When `--reasoning-parser` is enabled and a request uses both thinking (`chat_template_kwargs.thinking=true` / `require_reasoning=true`) and structured output (`response_format: json_object` or `json_schema`):

1. **JSON constraint runs too early** — The inner xgrammar JSON schema is applied from the first generated token, instead of only after `</think>` (the thinking phase). The model is steered to emit a complete JSON object immediately after the thinking prefix in the prompt.

2. **No separate content in the API response** — The full completion (often a single JSON blob) lands in `reasoning_content`; `message.content` is null or empty.

3. **Generation may stop after one JSON object** — Once xgrammar accepts a complete JSON object, constrained decoding can terminate; there is no second phase that produces a separate content field.

#### Related Changes

Fixed in [ReasonerGrammarObject(maybe_init_reasoning)](https://github.com/shinemo-ai/sglang/blob/0590737450afd8ef0577ed1a7c232cb116208a70/python/sglang/srt/constrained/reasoner_grammar_backend.py#L81).
