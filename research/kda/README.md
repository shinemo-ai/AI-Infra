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

其中，softplus(x) $=\ln(1+e^{x})$
