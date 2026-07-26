# Module 5: Distributed Parallelism and Multi-GPU Orchestration

As frontier LLMs scale to hundreds of billions of parameters (e.g., Llama 3 405B, DeepSeek-V3 671B), single-GPU memory and compute capacity are easily exceeded. High-throughput serving engines like vLLM partition model execution across multiple GPUs and multi-node clusters using **Distributed Parallelism**.

This module explores the four fundamental dimensions of distributed LLM serving: **Tensor Parallelism (TP)**, **Pipeline Parallelism (PP)**, **Context Parallelism (CP)**, and **Expert Parallelism (EP)** for Mixture-of-Experts (MoE) models.

---

## Part 1: Distributed Serving Taxonomy and Interconnect Constraints

To partition a model effectively, systems engineers match distributed parallelism strategies to the physical **hardware interconnect bandwidth** of the host cluster.

```mermaid
flowchart TD
    NODE["Single HGX Server Node (8 x H100 GPUs)"] --> NVLINK["🔥 NVLink 4 / NVSwitch Interconnect<br>Bandwidth: 900 GB/s per GPU (Bidirectional)"]
    CLUSTER["Multi-Node Cluster (Multiple Server Racks)"] --> IB["🚀 InfiniBand NDR / RoCE Network<br>Bandwidth: 400 Gbps (50 GB/s per node)"]

    NVLINK --> TP_SUITE["Use Tensor Parallelism (TP)<br>Requires High Intra-Node Bandwidth (All-Reduce per Layer)"]
    IB --> PP_SUITE["Use Pipeline Parallelism (PP) and Expert Parallelism (EP)<br>Tolerates Inter-Node Network Latency (Point-to-Point Transfers)"]
```

### 1.1 The Interconnect Bandwidth Gap

| Interconnect Level | Physical Technology | Bidirectional Bandwidth | Latency | Optimal Parallelism Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Intra-Node (SXM)** | NVIDIA NVLink 4 / NVSwitch | **$900 \text{ GB/s}$ per GPU** | $< 1 \ \mu\text{s}$ | **Tensor Parallelism (TP)** |
| **Intra-Node (PCIe)** | PCIe Gen5 x16 Bus | $64 \text{ GB/s}$ | $\sim 2 \dots 3 \ \mu\text{s}$ | Small Tensor Parallelism ($\text{TP} \le 4$) |
| **Inter-Node (RDMA)** | InfiniBand NDR / RoCE v2 | $50 \text{ GB/s}$ ($400 \text{ Gbps}$) | $\sim 5 \dots 10 \ \mu\text{s}$ | **Pipeline Parallelism (PP) & Expert Parallelism (EP)** |

---

### 1.2 Summary of Distributed Parallelism Dimensions

1. **Tensor Parallelism (TP)**: Splits individual weight matrices ($W_Q, W_K, W_V, W_O, W_{\text{Gate}}, W_{\text{Up}}, W_{\text{Down}}$) across GPUs inside a single node. Requires an **All-Reduce** communication step twice per Transformer layer.
2. **Pipeline Parallelism (PP)**: Partitions sequential Transformer layers across GPUs or server nodes (e.g., GPU 0 holds Layers 1-20, GPU 1 holds Layers 21-40). Communicates activation tensors via point-to-point transfers.
3. **Context Parallelism (CP)**: Partitions the sequence dimension (context length) across GPUs for massive $100K+$ context windows, using Ring-Attention or All-to-All communication.
4. **Expert Parallelism (EP)**: Distributes sparse Mixture-of-Experts (MoE) expert layers across GPUs, routing tokens dynamically via **All-to-All** communication primitives.

---

## Part 2: Tensor Parallelism (TP) Mechanics in vLLM

Tensor Parallelism in vLLM follows the **Shoeybi et al. (Megatron-LM)** matrix column/row splitting paradigm, enabling linear tensor operations to execute in parallel without intermediate communication overhead until output aggregation.

```mermaid
flowchart TD
    IN["Input Activation X"] --> COL_PAR["ColumnParallelLinear (W_Q, W_K, W_V, W_Gate, W_Up)<br>Splits Weight Matrix Column-Wise across GPUs"]
    COL_PAR --> ROW_PAR["RowParallelLinear (W_O, W_Down)<br>Splits Weight Matrix Row-Wise across GPUs"]
    
    subgraph ALL_REDUCE ["⚡ Custom NVLink All-Reduce Kernel"]
        ROW_PAR --> AR["vllm._C.custom_ar All-Reduce<br>Sum Partial Outputs across GPUs via Shared NVLink Memory"]
    end

    AR --> OUT["Final Aggregated Activation Tensor Y"]
```

### 2.1 ColumnParallelLinear and RowParallelLinear

#### 1. ColumnParallelLinear ($W_Q, W_K, W_V$ and $W_{\text{Gate}}, W_{\text{Up}}$)
A weight matrix $W \in \mathbb{R}^{d \times d_{\text{out}}}$ is split along its **columns** across $N_{\text{TP}}$ GPUs:

$$
W = \begin{bmatrix} W_1 & W_2 & \dots & W_{N_{\text{TP}}} \end{bmatrix}
$$

Each GPU $i$ receives a slice $W_i$ and multiplies against the input activation matrix $X$:

$$
Y_i = X @ W_i
$$

Because $Y_i$ is a column slice of the full output $Y = \begin{bmatrix} Y_1 & Y_2 & \dots & Y_{N_{\text{TP}}} \end{bmatrix}$, **no cross-GPU communication is needed** after a ColumnParallelLinear layer!

#### 2. RowParallelLinear ($W_O$ and $W_{\text{Down}}$)
A weight matrix $W \in \mathbb{R}^{d_{\text{in}} \times d}$ is split along its **rows** across $N_{\text{TP}}$ GPUs:

$$
W = \begin{bmatrix} W_1 \\ W_2 \\ \vdots \\ W_{N_{\text{TP}}} \end{bmatrix}
$$

Input activation $X$ (which is already split column-wise as $\begin{bmatrix} X_1 & X_2 & \dots & X_{N_{\text{TP}}} \end{bmatrix}$) multiplies against $W_i$:

$$
Y_i = X_i @ W_i
$$

To recover the true mathematical output $Y = X @ W$, partial outputs must be summed across all GPUs:

$$
Y = \sum_{i=1}^{N_{\text{TP}}} Y_i = \text{All-Reduce-Sum}(Y_i)
$$

---

### 2.2 Custom NVLink All-Reduce Kernels (`vllm._C.custom_ar`)

In PyTorch, standard `torch.distributed.all_reduce` invokes NCCL, which incurs kernel launch overhead and IPC synchronization latency ($\sim 10 \dots 15 \ \mu\text{s}$).

To eliminate NCCL host overhead for intra-node NVLink transfers, vLLM implements custom CUDA All-Reduce kernels (`vllm._C.custom_ar`):

1. **Shared NVLink IPC Buffer**: During engine initialization, vLLM registers static POSIX shared memory pointers across all intra-node GPUs over NVLink.
2. **Single-Kernel All-Reduce**: When `RowParallelLinear` completes, a single custom CUDA kernel executes across all GPUs simultaneously, reading partial tensors directly from peer GPU HBM over NVLink and performing the summation in-place.
3. **Latency Reduction**: Reduces All-Reduce communication latency from $15 \ \mu\text{s}$ down to **$< 2.5 \ \mu\text{s}$**, maximizing Tensor Core compute efficiency!

---

## Part 3: Pipeline Parallelism (PP) and Context Parallelism (CP)

When a model's memory footprint exceeds a single node's NVLink capacity (e.g. Llama 3 405B requiring $810 \text{ GB}$ in FP16), **Pipeline Parallelism (PP)** partitions layers across multiple node boundaries.

### 3.1 Pipeline Parallelism Mechanics

Pipeline Parallelism partitions the $L$ Transformer layers into contiguous stage groups across $N_{\text{PP}}$ GPUs/nodes (e.g., $N_{\text{PP}} = 4$ for an 80-layer model $\implies 20$ layers per stage).

```mermaid
flowchart LR
    subgraph STAGE0 ["Stage 0 (GPU 0: Layers 1..20)"]
        S0_P["Prefill / Decode Forward Pass"]
    end

    subgraph STAGE1 ["Stage 1 (GPU 1: Layers 21..40)"]
        S1_P["Prefill / Decode Forward Pass"]
    end

    subgraph STAGE2 ["Stage 2 (GPU 2: Layers 41..60)"]
        S2_P["Prefill / Decode Forward Pass"]
    end

    subgraph STAGE3 ["Stage 3 (GPU 3: Layers 61..80)"]
        S3_P["Prefill / Decode Forward Pass"]
    end

    S0_P -->|"Point-to-Point Transfer (P2P)"| S1_P
    S1_P -->|"Point-to-Point Transfer (P2P)"| S2_P
    S2_P -->|"Point-to-Point Transfer (P2P)"| S3_P
```

#### Communication Efficiency
Unlike Tensor Parallelism (which requires 2 All-Reduces per layer), Pipeline Parallelism requires only **1 Point-to-Point (P2P) activation transfer between stage boundaries**. This minimal network payload allows PP to run efficiently over inter-node InfiniBand / RoCE links ($50 \text{ GB/s}$).

---

### 3.2 Context Parallelism (CP) and Ring-Attention

For extreme long-context workloads ($S > 100,000$ tokens), a single GPU cannot store the physical KV cache blocks for the entire sequence even with PagedAttention.

**Context Parallelism (CP)** partitions the sequence length dimension $S$ across $N_{\text{CP}}$ GPUs using **Ring-Attention** (Liu et al.):

```mermaid
flowchart TD
    subgraph RING ["🔄 Ring-Attention Communication Loop"]
        GPU0["GPU 0 (Tokens 0..25K)<br>Holds Q_0, K_0, V_0"] -->|"Pass K, V Block"| GPU1["GPU 1 (Tokens 25K..50K)<br>Holds Q_1, K_1, V_1"]
        GPU1 -->|"Pass K, V Block"| GPU2["GPU 2 (Tokens 50K..75K)<br>Holds Q_2, K_2, V_2"]
        GPU2 -->|"Pass K, V Block"| GPU3["GPU 3 (Tokens 75K..100K)<br>Holds Q_3, K_3, V_3"]
        GPU3 -->|"Pass K, V Block"| GPU0
    end
```

1. Each GPU $i$ holds a slice of Query tokens $Q_i$ and a slice of Key/Value tokens $K_i, V_i$.
2. While GPU $i$ computes partial SRAM attention between $Q_i$ and local $K_i, V_i$, it asynchronously streams its $K_i, V_i$ blocks to the next GPU in a ring topology.
3. Over $N_{\text{CP}}$ ring steps, every GPU evaluates full attention over all sequence tiles without any single GPU holding the complete KV cache in memory!

---

## Part 4: Expert Parallelism (EP) for Mixture-of-Experts (MoE)

Modern frontier models like **DeepSeek-V3 / DeepSeek-R1** (671 Billion total parameters, 37 Billion active parameters per token) and **Mixtral 8x22B** utilize **Mixture-of-Experts (MoE)** architectures.

In an MoE Transformer layer, dense Feed-Forward Networks (FFNs) are replaced by a **Router / Gating Network** and $N_{\text{experts}}$ parallel Expert networks.

### 4.1 MoE Top-k Routing Mechanics

For each token $x$, a router network computes softmax probabilities over all experts and selects the **Top-$k$** highest scoring experts (e.g. top-2 for Mixtral, top-8 for DeepSeek-V3):

$$
g(x) = \text{TopK}\left(\text{Softmax}(x @ W_{\text{gate}})\right)
$$

$$
y = \sum_{i \in \text{TopK}} g(x)_i \cdot \text{Expert}_i(x)
$$

```mermaid
flowchart TD
    X["Input Token Vector x"] --> ROUTER["Gating Router Network (W_gate)"]
    ROUTER --> TOPK["Select Top-k Experts (e.g. Top-2 or Top-8)"]
    
    TOPK --> E1["Expert 1 (FFN)"]
    TOPK --> E4["Expert 4 (FFN)"]
    
    E1 --> SUM["Weighted Sum of Expert Outputs"]
    E4 --> SUM
    SUM --> Y["Final Output Vector y"]
```

---

### 4.2 Expert Parallelism (EP) and Fused MoE Kernels

In large MoE models with 64 or 256 total experts (e.g. DeepSeek-V3 with 256 routing experts across 8 nodes):

1. **Expert Distribution (EP)**: Experts are distributed evenly across GPUs ($N_{\text{experts}} / N_{\text{GPUs}}$ experts per GPU).
2. **All-to-All Communication (`all_to_all_single`)**:
   - **Dispatch Pass**: Tokens are routed to the specific GPUs holding their assigned Top-$k$ experts using an `All-to-All` network transfer.
   - **Combine Pass**: After expert GPUs compute FFN math, output vectors are transferred back to the origin GPUs via a second `All-to-All` network transfer.

```mermaid
flowchart LR
    subgraph DISPATCH ["1. Dispatch All-to-All"]
        G0_TOK["GPU 0 Tokens"] -->|"Route to Assigned Experts"| EP_GPU["Expert GPUs (EP 0..N)"]
    end

    subgraph COMPUTE ["2. Fused MoE CUDA Computation"]
        EP_GPU --> FUSED["vllm._C.fused_moe<br>Groups tokens by expert and executes GEMM in SRAM"]
    end

    subgraph COMBINE ["3. Combine All-to-All"]
        FUSED -->|"Return Result Vectors"| G0_OUT["GPU 0 Output Aggregation"]
    end
```

#### Fused MoE CUDA Kernels (`vllm._C.fused_moe`)
Executing individual GEMMs for scattered token subsets causes massive CPU launch overhead and GPU fragmentation. vLLM implements custom **Fused MoE CUDA Kernels (`fused_moe`)**:

- **Token Sorting**: The kernel sorts and groups all tokens in SRAM by their target expert ID.
- **Batched GEMM Execution**: Executes a single grouped GEMM CUDA launch that processes all expert workloads simultaneously without returning to the CPU, maximizing Tensor Core compute efficiency!

---

## Summary: Parallelism Strategy Decision Matrix

| Model Architecture / Size | Recommended Parallelism Setup | Primary Communication Primitives | Target Hardware Topology |
| :--- | :--- | :--- | :--- |
| **Medium Models (7B - 70B)** | $\text{TP} = 2 \dots 8$, $\text{PP} = 1$ | All-Reduce (`vllm._C.custom_ar`) | Single HGX Node (NVLink) |
| **Large Dense Models (405B)** | $\text{TP} = 8$, $\text{PP} = 8 \dots 16$ | All-Reduce (Intra-Node) + P2P (Inter-Node) | Multi-Node NVLink + InfiniBand |
| **Extreme Long Context (100K+)** | $\text{TP} = 8$, $\text{CP} = 4 \dots 8$ | Ring-Attention (Async P2P / All-to-All) | Multi-Node High-Bandwidth Network |
| **Massive MoE (DeepSeek / Mixtral)** | $\text{TP} = 8$, $\text{EP} = 8 \dots 64$ | All-to-All (`all_to_all_single`) + Fused MoE | Multi-Node Cluster with InfiniBand RDMA |

With **Module 5** complete, we advance to **Module 6: Production Deployment and Cloud Orchestration (`06_deployment_and_orchestration.md`)**, where we explore OpenAI API server architecture, Ray Core actor clusters, Kubernetes deployments (NVIDIA GPU Operator, `/dev/shm` IPC sizing), and KEDA autoscaling.
