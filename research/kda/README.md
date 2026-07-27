# Kimi Delta Attention (KDA)

Kimi Delta Attention 模型结构

**Kimi Delta Attention (KDA)** 是 Kimi 系列大模型在结构上最重要的创新之一。它由 Gated Delta Attention (GDA) 演化而来，二者的核心差异在于**衰减（decay）方式**：

- **GDA**：衰减因子是**标量**，所有通道共享同一个衰减系数，控制粒度较粗；
- **KDA**：衰减因子升级为**向量**，并构成对角矩阵 $\mathrm{Diag}(\alpha_t)$，从而实现**逐通道、细粒度**的门控。

## 状态更新公式

GDA 的状态更新：

$$
S_t = S_{t-1} \big( \alpha_t (I - \beta_t k_t k_t^{T}) \big) + \beta_t v_t k_t^{T}
$$

KDA 的状态更新：

$$
S_t = (I - \beta_t k_t k_t^{T}) \ \mathrm{Diag}(\alpha_t) \ S_{t-1} + \beta_t k_t v_t^{T} \in \mathbb{R}^{d_k \times d_v}
$$

$$
o_t = S_t^{T} q_t \in \mathbb{R}^{d_v}
$$

## 模型结构与计算路径

论文结构图中，KDA 各分量的计算可概括为：

$$
q_t^{h},\ k_t^{h} = \mathrm{L2Norm}\big(\mathrm{Swish}(\mathrm{ShortConv}(W_{q/k}^{h} x_t))\big) \in \mathbb{R}^{d_k}
$$

$$
v_t^{h} = \mathrm{Swish}(\mathrm{ShortConv}(W_v^{h} x_t)) \in \mathbb{R}^{d_v}
$$

$$
\alpha_t^{h} = f(W_{\alpha}^{\uparrow} W_{\alpha}^{\downarrow} x_t) \in [0, 1]^{d_k}
$$

$$
\beta_t^{h} = \mathrm{Sigmoid}(W_{\beta}^{h} x_t) \in [0, 1]
$$

对应到结构图右侧： $q$ / $k$ 经 Linear → Conv → Swish → L2 Norm；v 经 Linear → Conv → Swish； $\alpha$ / $\beta$ 等门控量由低秩 / 线性投影再经激活得到。

## 实现要点

从公式与结构可以看出，KDA 使用了：

- 一维卷积（ShortConv）
- Swish / Sigmoid 激活
- L2 归一化

开源推理框架的实现也基本按上述路径落地。需要注意的是，论文中的 $f(\cdot)$ 被称为 **decay function（衰减函数）**，刻画在更新状态时对旧状态的保留程度。

在 SGLang 中，该衰减函数有两种写法：


| 写法           | 说明                                |
| ------------ | --------------------------------- |
| **standard** | Kimi 实际采用的写法，与 Mamba / GDA 经典形式一致 |
| **safe**     | 数值更稳健的变体                          |


当前 Kimi 使用的是 **standard** 写法。

### SGLang 调用链路

```text
KimiDeltaAttention.forward
└── self.attn(...)                              # RadixLinearAttention
    └── get_attn_backend().forward              # HybridLinearAttnBackend
        ├── decode → KDAAttnBackend.forward_decode
        │   ├── causal_conv1d_update            # ShortConv
        │   └── kernel_dispatcher.packed_decode / decode
        │       └── TritonKDAKernel
        │           ├── fused_recurrent_kda_packed_decode
        │           │                               # fla/fused_recurrent.py
        │           └── fused_sigmoid_gating_delta_rule_update
        │                                           # fla/fused_sigmoid_gating_recurrent.py
        │
        └── prefill/extend → KDAAttnBackend.forward_extend
            ├── causal_conv1d_fn                # ShortConv
            └── kernel_dispatcher.extend
                └── TritonKDAKernel.extend
                    └── chunk_kda(...)          # fla/kda.py（论文 chunk 公式主入口）
```

在`RadixLinearAttention`里如果是普通eager prefill会直接进入else分支：

```python
get_attn_backend().forward(
    layer=self,
    forward_batch=forward_batch,
    mixed_qkv=mixed_qkv,
    a=a,
    b=b,
)
```

`get_attn_backend()` 拿到的是 `HybridLinearAttnBackend`，先调它的 `forward`，再按层分流：

```text
get_attn_backend().forward(...)          # HybridLinearAttnBackend
  ├─ full attn 层 → full_attn_backend
  └─ KDA 层 → linear_attn_backend.forward_decode / forward_extend
                    └─ 这里才是 KDAAttnBackend
```
 
`KDAAttnBackend` 里核心的算子实现使用triton kernel，即`KDAKernelDispatcher`，当然它只是上层的封装，实际定义了`triton_kernel = TritonKDAKernel()`。如果是decode就调用`TritonKDAKernel`的decode方法，prefill就调用extend方法。
再细看extend：
```python
def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        return chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=ssm_states,
            initial_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=query_start_loc,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
        )
```
其中`chunk_kda`来源于`python/sglang/srt/layers/attention/fla/kda.py`，即
```python
def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    initial_state_indices: torch.Tensor = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: Optional[float] = None,
    **kwargs,
):
    if scale is None:
        scale = k.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q.contiguous())
        k = l2norm_fwd(k.contiguous())

    o = chunk_kda_fwd(
        q=q,
        k=k,
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=scale,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )
    return o
```
核心是`chunk_kda_fwd`，先看衰减门控 $g$ 的实现分两种情况：
```python
if A_log is not None:
    # Fused: gate activation + chunk-local cumsum in one kernel.
    # g is raw gate (before activation); A_log, dt_bias drive the activation.
    g = kda_gate_chunk_cumsum(
        g,
        A_log=A_log,
        chunk_size=chunk_size,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        lower_bound=lower_bound,
    )
else:
    # g is already gate-activated by caller; just do cumsum.
    g = chunk_local_cumsum(
        g,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
```
`if A_log is not None`是Kimi默认路径，`kda_gate_chunk_cumsum`其实就是一次kernel做完两件事情，因为这种情况下`chunk_kda_fwd`传入的 $g$ 其实就是 $W_{\alpha}^{\uparrow} W_{\alpha}^{\downarrow} x_t$ 。`kda_gate_chunk_cumsum`底层核心调用的是`kda_gate_chunk_cumsum_vector_kernel`这个triton kernel，其实主要做了以下运算：

$$
s_t = W_{\alpha}^{\uparrow} W_{\alpha}^{\downarrow} x_t
$$

$$
g_t = -\exp(A_{\log}) \cdot \mathrm{softplus}(s_t + b_{bias})
$$

其中，softplus(x) $=\ln(1+e^{x})$ 。 然后调用`tl.cumsum`对 $g_t$ 做累加得到 $G_t$ ，因为在`chunk_kda_fwd_kernel_intra_token_parallel`需要使用 $G_t$ 来计算 $e^{G_t}$ 。

ShortConv实际是因果1D卷积，且卷积核的大小为4（short_conv_kernel_size），步长为1，即每个位置用长度为4的窗口看当前及过去共4个token（过去3个token + 当前1个token），窗口每次向前滑1步。正是因为卷积步长为1，所以经过ShortConv后的张量维度与输入保持一致，代码里没有明确指定stride，具体可见`python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py`，在`_causal_conv1d_fwd_kernel`里：
```python
for idx_token in range(segment_len):
    ...
    matrix_x = col0
    ...
    elif KERNEL_WIDTH == 4:
        col0 = col1
        col1 = col2
        col2 = matrix_x
```
其实就是在当前拍卷积只看col0、col1、col2和matrix，再下一拍就是上一拍的col1（t-3）、col2（t-2）、matrix（t-1），再加上下一拍的token t。

对于ShortConv，如果序列长度为 $T$ ，因果卷积就要做 $T$ 次卷积，每个token位置一次，不够卷积核大小左侧padding，例如：

| 位置 | 窗口（不足补 0） |
|------|------------------|
| t=0 | `[0, 0, 0, x0]` |
| t=1 | `[0, 0, x0, x1]` |
| t=2 | `[0, x0, x1, x2]` |
| t=3 | `[x0, x1, x2, x3]` |
| t=4 | `[x1, x2, x3, x4]` |
| t=5 | `[x2, x3, x4, x5]` |

如果开启`radix cache`即需要前缀缓存命中，`conv_state`需要保存下来。在开启chunk prefill时，单次forward（一个chunk）内部：ShortConv（kernel=4）的滑窗状态保留在寄存器中，每处理完一个token只滚动保留最近3个投影后的输入特征时间步（即 $\x_{t-3}$ , $\x_{t-2}$, $\x_{t-1}$ ），chunk边界再把这3个输入特征写回GPU上的`conv_state`。

上述均在`_causal_conv1d_fwd_kernel`里，chunk内写回可见：
```python
for idx_token in range(segment_len):
    ...
    matrix_x = col0
    ...
    elif KERNEL_WIDTH == 4:
        col0 = col1
        col1 = col2
        col2 = matrix_x
```
chunk/forward写回可见：
```python
# STEP 2:
# here prepare data for updating conv_state
if (
    state_len <= seqlen
):  # SMALL_CACHE=True (only move part of 'x' into conv_state cache)
    # just read from 'x'
    # copy 'x' data to conv_state
    # load only 'x' data (and set 0 before 'x' if seqlen < state_len)
    idx_tokens_last = (seqlen - state_len) + tl.arange(
        0, NP2_STATELEN
    )  # [BLOCK_M]
    x_ptrs = (
        x_ptr
        + ((sequence_start_index + idx_tokens_last) * stride_x_token)[:, None]
        + (idx_feats * stride_x_dim)[None, :]
    )  # [BLOCK_M,BLOCK_N,]
    mask_x = (
        (idx_tokens_last >= 0)[:, None]
        & (idx_tokens_last < seqlen)[:, None]
        & (idx_feats < dim)[None, :]
    )  # token-index  # token-index  # feature-index
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
    idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]
    conv_states_ptrs_target = (
        conv_states_base[None, :]
        + (idx_tokens_conv * stride_conv_state_tok)[:, None]
    )

    mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.debug_barrier()  #  NOTE: use this due to bug in Triton compiler
    tl.store(conv_states_ptrs_target, new_conv_state, mask)
```