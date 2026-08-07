# 附录: 主词汇表与术语权威参考

本权威参考文档汇编了全书 **vLLM 核心架构 (`模块 1 至 6`)** 所涉及的所有数学、架构、硬件、分布式系统以及云原生编排术语。每个条目均给出了精确的物理定义、数学公式推导以及系统工程作用。

---

## 1. 架构与数学术语

### `RoPE` (Rotary Position Embedding - 旋转位置编码)
- **定义**: 一种位置编码机制 (由 Su 等人于 2021 年提出)，通过在点积注意力计算前将旋转矩阵直接乘至 Query (`Q`) 和 Key (`K`) 向量，将 Token 顺序信息注入 Transformer 模型中。
- **数学机制**:
  `RoPE` 并不像传统的绝对位置编码那样将位置向量 $P_m$ 加到词嵌入上 ($X + P_m$)，而是将注意力 Head 内部的相邻维度 (`head_dim = 128`) 配对为 2D 旋转平面 ($64 \text{ 对}$)。对于位于序列索引 $m$ 处的 Token，每个 2D 平面乘以一个 2D 旋转矩阵 $R_{m,\theta_i}$：

$$
R_{m, \theta_i} = \begin{bmatrix} \cos(m \cdot \theta_i) & -\sin(m \cdot \theta_i) \\ \sin(m \cdot \theta_i) & \cos(m \cdot \theta_i) \end{bmatrix}
$$

  在计算位置 $m$ 的 Query ($Q_m$) 与位置 $n$ 的 Key ($K_n$) 之间的点积时，正交旋转代数特征保证了最终的内积**严格取决于两个 Token 之间的相对距离 ($m - n$)**：

$$
\text{Score} = (R_m @ Q_m) @ (R_n @ K_n)^T = Q_m @ R_{m-n} @ K_n^T
$$

- **架构意义**: 消除了绝对位置偏差，保持了基础语义向量的模长不变，并支持平滑的长上下文外推 (`例如通过 NTK-aware 缩放或 YaRN 将上下文从 8K 扩展到 128K`)。
- **$R_m$ 的精确矩阵维度 (`[128, 128]` 块对角矩阵)**:
  `RoPE` 不会跨越不同的注意力 Head (`num_kv_heads = 8`) 进行旋转，因为每个 Head 编码正交的语义领域。相反，`RoPE` 严格在每个 Head 内部的 `128` 个特征坐标 (`Axis m = 0..127`) 上运行，将其划分为 64 个 2D 对。写成作用在 `[128]` 维 Head 向量上的统一算子时，$R_m$ 是一个**包含 64 个 `2x2` 旋转块的 `[128, 128]` 块对角矩阵**。因此，所有 `64` 个 Query Head 和 `8` 个 Key Head 都独立经历这个 `[128, 128]` 旋转。此外，`KV` Cache 中历史的 Key ($k_0, k_1, \dots, k_{n-1}$) 在最初生成时就已经旋转好并永久保存，无需重新旋转。

- **$R_m$ 的对称性与确定性 (`Q/K 对称性与 Theta 频率层级`)**:
    1. **`Q/K` 对称性**: 在序列位置 $m$ 处，完全相同的旋转矩阵 $R_m$ 同时作用于 Query ($Q_m$) 和 Key ($K_m$)。
    2. **通用 Head 应用**: $R_m$ 相同地应用于所有 `64` 个 Query Head 与 `8` 个 Key Head。
    3. **严格确定性**: $R_m$ 完全独立于 Token 的具体文本或 Activation 值 (`"苹果"` 与 `"狗"` 在位置 $m$ 处产生完全相同的 $R_m$)。它完全由整数序列索引 $m$ (`0, 1, 2, ...`) 和几何角频率层级 $\theta_i = 10000^{-2i / d_{\text{head}}}$ 决定。
- **执行生命周期管线 (`Embedding vs. RoPE`)**:
  初始 Token 词表查表 ($W_{\text{Embed}}$) 与 `RoPE` 旋转运行在两个不同的执行阶段：
    1. **Step 0 (`Embedding 查表`)**: 将 Token ID 转换为未旋转的基础张量 $X_0$ (`[b, s, d]`)，**完全不含位置信息**。
    2. **Step 1 (`线性投影`)**: 在每一层内部，$X$ 乘以 $W_Q$ 和 $W_K$ 产生未旋转的 $Q$ 和 $K$ 张量。
    3. **Step 2 (`RoPE 旋转`)**: 严格在 `Q/K` 线性投影**之后**、点积注意力计算**之前**，旋转矩阵 ($R_m$) 乘以 $Q$ 和 $K$ ($Q_{\text{rotated}} = R_m @ Q_{\text{unrotated}}$)。

### `SwiGLU` (Swish-Gated Linear Unit - Swish 门控线性单元)
- **定义**: 一种门控非线性激活架构 (由 Noam Shazeer 提出)，在现代前沿模型 (`Llama 3, Mistral, DeepSeek`) 中取代了经典的 `ReLU` 前馈神经网络 (`FFN`)。
- **数学机制**:
  每层需要三个投影矩阵 (`W_Gate`, `W_Up`, `W_Down`)：
  $$
\text{FFN_Output} = \left( \text{SiLU}(H @ W_{\text{Gate}}) \odot (H @ W_{\text{Up}}) \right) @ W_{\text{Down}}
$$
  其中 $\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$ 充当连续可导的门控调制曲线，`(*)` 表示逐元素 Hadamard 乘法。

### `MHA` (Multi-Head Attention - 多头注意力)
- **定义**: 原始标准注意力架构 (`Vaswani et al., 2017`)，每个 Query Head (`h_q`) 都配有一个独立的 Key (`h_k`) 和 Value (`h_v`) 投影 Head (`h_q : h_kv = 1 : 1`)。
- **权衡**: 最大化了特征表达能力，但在长上下文自回归解码期间由于巨大的 `KV` Cache 占用导致显存极大受限。

### `MQA` (Multi-Query Attention - 多查询注意力)
- **定义**: 一种注意力变体 (`Shazeer, 2019`)，所有 `num_q_heads` (`例如 64 个`) 共享**单个 Key 和 Value Head (`h_kv = 1`)**。
- **权衡**: 极大地压缩了 `KV` Cache 显存占用 (`h_q : 1` 压缩比)，但会导致特征表达碰撞和容量饱和，在复杂的推理任务上表现出精度下降。

### `GQA` (Grouped-Query Attention - 分组查询注意力)
- **定义**: 帕累托最优的注意力架构 (`Ainslie et al., 2023`)，将 Query Head 划分为不同的组，每组共享单个 Key/Value Head (`例如 Llama 3 70B 为 64 个 Q Head 使用 8 个 KV Head，比例为 8:1`)。
- **权衡**: 实现了接近 `MHA` 的表达质量，同时将 `KV` Cache 显存占用压缩了 `8 倍`。

### `MLA` (Multi-Head Latent Attention - 多头潜隐注意力)
- **定义**: 一种低秩联合 `KV` 压缩架构 (`DeepSeek-V2/V3/R1`)，旨在超大规模前沿模型中大幅压缩显存占用而不牺牲推理精度。
- **数学机制**:
  将输入 Activation `H` 投影为一个低秩潜隐向量 `c_kv` (`d_c = 512`)，作为唯一的缓存载荷。为了防止位置旋转解偶纠缠，`RoPE` 被解耦到一个独立的 64 维位置 Key 向量 (`k_pe`)：
  $$
\text{每个 Token 缓存的总向量} = c_{kv} \ (512 \text{ 维}) + k_{pe} \ (64 \text{ 维}) = \mathbf{576 \text{ 个浮点数}}
$$
  在注意力计算时，完整的 `K_content` 与 `V` 向量在 GPU 片上 `SRAM` 中动态还原 (`c_kv @ W_UK` 与 `c_kv @ W_UV`)，或直接吸收至 Query 权重矩阵 (`矩阵吸收`).

### `MoE` (Mixture of Experts - 混合专家模型)
- **定义**: 一种稀疏神经网络架构，通过用动态 Router (门控网络) 管理的多个并行专家 `FFN` 取代稠密 `FFN` 层，从而将模型总参数量与单 Token 激活算力 (`FLOPs`) 解耦。
- **机制**: 对于每个 Token，Router 计算对所有专家的相似度 Logits ($H @ W_{\text{Router}}$)，选择 Logits 最高的 `Top-K` 个专家 (`例如 Mixtral 8x7B 挑选 Top-2`)，并在选定子集中应用 `Softmax` 进行加权求和。

---

## 2. 线性代数与执行算子术语

### `GEMM` (General Matrix-Matrix Multiplication - 通用矩阵乘法)
- **定义**: 控制计算受限神经网络层的主要线性代数算子 ($C = A @ B + D$)。
- **推理语境**: 占据 **Prefill 预填充阶段**，所有 Prompt Token ($s_{\text{prompt}} > 1$) 同时被并行计算，在极高的算术强度下打满 GPU Tensor Core。

### `GEMV` (General Matrix-Vector Multiplication - 通用矩阵-向量乘法)
- **定义**: 矩阵乘以单个向量的线性代数算子 ($c = A @ b$)。
- **推理语境**: 占据 **Decode 解码阶段** ($batch_size = 1$)，每步仅计算一个新生成的 Token ($seq_len = 1$)。`GEMV` 处于严重的显存带宽受限区域 ($I_{\text{decode}} \approx 1.0 \dots 2.0 \text{ FLOPs/Byte}$)。

### `BMM` (Batched Matrix Multiplication - 批量矩阵乘法)
- **定义**: 跨高维张量的并行矩阵乘法算子 (`[b, h, m, k] @ [b, h, k, n] = [b, h, m, n]`)。
- **机制**: 严格在最后两个维度上执行计算，将其余前导维度视为独立的并行 Batch 索引分发到各个 GPU SM 核心。

---

## 3. 硬件与内存层级术语

### `HBM` (High-Bandwidth Memory - 高带宽显存)
- **定义**: 通过超宽硅中介层直接连接到 GPU 芯片的片外 DRAM 堆栈 (`例如 NVIDIA H100/H200 上的 80 GB 或 192 GB 容量`)。
- **作用与带宽**: 作为保存模型权重和静态 `KV` Cache Block 的主全局内存池。在 H100 SXM5 上提供 `3.35 TB/sec` 的带宽 (`比片上 L2 Cache 慢 4.5 倍`)。

### `L2 Cache` (二级静态缓存)
- **定义**: 位于 GPU 芯片上的片上静态随机存取内存 (`SRAM`)，由所有流多处理器 (`SM`) 共享 (`H100 SXM5 上为 50 MB`)。
- **作用与带宽**: 作为片外 `HBM` 与单个 SM 核心之间的超高速暂存缓冲区 (`12 到 15 TB/sec` 带宽)。

### `SM` (Streaming Multiprocessor) 与 寄存器 / `L1 Cache`
- **定义**: GPU 内部的独立物理计算核心 (`NVIDIA H100 包含 132 个 SM`)。每个 SM 包含本地 `SRAM` (`寄存器和 ~256 KB L1/共享内存`)，提供 `30 到 50+ TB/sec` 的极高带宽。
- **作用**: 通过专用的硬件计算单元 (`Tensor Cores`) 执行方形线程块 Tile (`例如 128x128 Tile`)。

### `Arithmetic Intensity` (算术强度 $I$) 与 Roofline 模型
- **定义**: 执行的总浮点运算次数 (`FLOPs`) 除以从 `HBM` 传输的总物理显存字节数 (`Bytes`) 的比值：
  $$
I = \frac{\text{执行的总 FLOPs}}{\text{从 HBM 传输的总字节数}}
$$
- **转折点 (`I_ridge`)**: 区分计算受限执行与显存带宽受限执行的精确阈值 ($\text{I_ridge} = \text{峰值 FLOPs} / \text{峰值 HBM 带宽}$，在 H100 上约为 `295 FLOPs/Byte`)。
- **对比评估规则**: 在对比不同 GPU (`A100 vs. H100 vs. MI300X`) 时，更高的 `I_ridge` **并不**意味着 GPU 更适合解码推理。由于 LLM 自回归解码处于严重的显存受限斜率 ($I \approx 1.0 \dots 2.0 \text{ FLOPs/Byte}$)，吞吐量完全由 `峰值 HBM 带宽` 决定。因此，**更低的 `I_ridge` 结合更高的 HBM 带宽** (`例如 AMD MI300X 拥有 245 FLOPs/Byte 与 5.3 TB/sec 带宽`) 在解码阶段远胜高 `I_ridge`。

---

## 4. 分布式系统与并行术语

### `EP` (Expert Parallelism - 专家并行)
- **定义**: `MoE` 架构的分布式切分策略，不同的物理 GPU 节点或设备保存互不重叠的专家权重表 (`例如 GPU 0 保存专家 0，GPU 7 保存专家 7`)。
- **执行机制**: 需要跨越高速互连网络 (`NVLink` 或 `RDMA`) 进行 `All-to-All` 集合通信，将 Token Activation 向量 ($H$) 分发到其指定的目标专家 GPU，并取回计算后的结果。

### `TP` (Tensor Parallelism - 张量并行)
- **定义**: 节点内分布式切分策略 (`Megatron-LM`)，将单个线性权重矩阵 (`W_Q, W_K, W_V, W_1, W_2`) 切分到同一服务器节点内的多块物理 GPU 上。
- **执行机制**: 执行按列或按行切分的矩阵变换，在每层 Transformer 之后需要通过 `NVLink / NVSwitch` 进行亚毫秒级的 `AllReduce` 同步。

### `PP` (Pipeline Parallelism - 流水线并行)
- **定义**: 跨节点分布式切分策略，将连续的 Transformer 层划分给序列上的物理服务器节点 (`例如 节点 0 运行第 1-20 层，节点 1 运行第 21-40 层`)。
- **执行机制**: 利用微批处理 (`1F1B 调度`) 在各个 Stage 之间重叠计算，降低流水线气泡 (Bubble) 延迟。

---

## 5. 服务引擎与分页内存术语

### `PagedAttention`
- **定义**: vLLM 借鉴操作系统虚拟内存分页机制打造的基石注意力 Kernel。它彻底解耦了逻辑 Token 序列顺序与物理连续 `HBM` 位置，使注意力分数能够跨越离散的 `16-token` 物理 Block 在 $\mathcal{O}(1)$ 查表时间内完成计算，且无显存碎片。

### `CoW` (Copy-on-Write - 写时复制)
- **定义**: vLLM 内部的 Block 管理机制，多个逻辑请求 (`例如共享系统提示词、Beam Search 分支、并行 best_of_n 采样`) 通过引用计数 (`ref_count > 1`) 指向完全相同的物理 `KV` Cache Block。仅在某个请求在生成过程中发生分歧时才进行物理复制。

### `APC` (Automatic Prefix Caching - 自动前缀缓存)
- **定义**: vLLM 内部基于哈希的 Block 管理算法，通过计算 Token 序列的密码学哈希值，识别并跨独立、异步的用户请求复用可重用的前缀 Block。

### `Block Table` (块表)
- **定义**: vLLM 中每个序列的元数据查找表，在 $\mathcal{O}(1)$ 时间内将逻辑 Block 索引 (`例如 逻辑 Block 1`) 转换为物理 `HBM` Block 指针 (`例如 物理 Block 28`)。同时追踪 `num_filled_tokens` 和 Block 引用计数 (`ref_count`)。

### `Online Softmax`
- **定义**: PagedAttention Kernel 中采用的增量在线 Softmax 算法，维护动态最大值 $m_i$ 与指数累加和 $l_i$，允许在片上 SRAM 寄存器中直接完成非连续物理 Block 的注意力归一化与 Value 累加，无需中间显存落盘。

### `Continuous Batching` (连续批处理 / 迭代级调度)
- **定义**: 一种动态调度范式 (`Orca, Yu et al., 2022`)，推理服务器在**每一个 Token 生成 Step (Iteration)** 上检查请求队列，立即弹出已完成的序列 (`EOS`) 并吸纳新请求 (`Waiting -> Running`)，消除了传统静态批处理高达 90% 的 GPU 闲置浪费。

### `Preemption` (抢占机制: Swap vs. Recomputation)
- **定义**: 当自回归解码期间 HBM 空闲块耗尽时触发的容错恢复机制。调度器将低优先级的 `Running` 序列降级为 `Swapped` (通过 PCIe 将物理 KV 块驱逐至 Host 内存) 或 `Waiting` (直接丢弃 KV 块以待后续重新计算)。

### `Internal Fragmentation` (内部碎片)
- **定义**: 分配的物理内存块大小超过了实际存储的有效载荷导致的显存浪费。在传统连续预分配中浪费率高达 60%-80%；vLLM 将内部碎片消除并严格隔离至序列尾部的最后一个未填满 Block (<4% 浪费)。

### `External Fragmentation` (外部碎片)
- **定义**: 未分配的空闲内存散落在物理地址空间中但互不连续，导致无法满足连续分配请求的现象。vLLM 通过 PagedAttention 允许非连续寻址，将外部碎片彻底降至 **0%**。

---

## 6. 云原生编排、存储与机密 AI 术语

### `LeaderWorkerSet` (`LWS`)
- **定义**: 专为多节点分布式 AI/LLM 工作负载设计的 Kubernetes 原生控制器 API。它将分布式模型副本建模为 1 个 Leader Pod (服务入口/Rank 0) 与 $N-1$ 个 Worker Pod (Rank $1 \dots N-1$)。
- **关键能力**: 原生支持 **原子 Gang 故障生命周期** (`RecreateGroupOnPodRestart`)，确保在多机 TP/PP 组中任意节点故障时全组同步重启，杜绝集群出现残缺死锁。

### `Hyperdisk ML`
- **定义**: Google Cloud 专为 AI 推理与训练设计的超高吞吐分布式块存储架构。
- **在 LLM 推理中的作用**: 支持以 **Read-Only Many (`ROX`)** 模式同时挂载给多达 **2,500 个 GKE Pod**，提供高达 **$1.2 \text{ TB/s}$ 的集群聚合读取吞吐量**。将 70B/405B 大模型的冷启动加载耗时从 15–30 分钟降至 10 秒以内。

### `Google Cloud AI Hypercomputer`
- **定义**: Google Cloud 的超级计算 AI 体系架构，深度整合了专用加速硬件 (TPU v5e/v5p/v6e Trillium、NVIDIA H100/H200/B200 A3/A4 实例)、超高速网络互联 (Titanium 卸载、RoCEv2、GPUDirect-TCPX/RDMA、NCCL FastSocket)、高吞吐存储 (Hyperdisk ML、GCS FUSE) 以及 GKE 编排 (LWS、KubeRay、Kueue、KEDA)。

### `Confidential Space` 与 `Confidential GKE`
- **定义**: Google Cloud 基于硬件隔离的可信执行环境 (TEE) 打造的零信任机密计算体系，依托机密虚拟机 (AMD SEV-SNP / Intel TDX) 与 **机密 GPU (NVIDIA H100 TEE APM 模式)**。
- **安全作用**: 确保物理机 Host OS 管理员、被攻破的 Hypervisor、云平台运维人员以及多租户邻居均无法窥探显存 (HBM)、内存 (RAM) 以及 PCIe 总线中的明文模型权重与用户数据。

### `Remote Attestation` (密码学远程证明)
- **定义**: 运行在 TEE 内的硬件安全模块 (vTPM 与 H100 安全处理器) 对系统的启动状态、固件、内核与容器镜像计算密码学哈希，并生成带硬件签名的 Quote。证明服务验证 Quote 后签发 OIDC Token，由 Cloud KMS 校验并下发对称解密密钥。

### `Disaggregated Serving` (Prefill 与 Decode 解耦服务)
- **定义**: 将计算密集型的 **Prefill 节点** (优化大 Batch GEMM 算力) 与显存带宽密集型的 **Decode 节点** (优化低延迟自回归解码) 在物理服务器级别分离，通过超低延迟 RDMA 跨网络直传物理 KV Cache Block 的先进推理服务拓扑。

### `Prefix-Cache-Aware Routing` (前缀缓存感知路由)
- **定义**: 智能 API 网关路由策略，对传入 Prompt 的公共前缀（系统提示词、长文档上下文）计算一致性哈希，将具有相同前缀特征的请求始终路由到同一个 vLLM Pod 副本，将自动前缀缓存 (APC) 命中率从 $\sim 15\%$ 提升至 $> 90\%$。
