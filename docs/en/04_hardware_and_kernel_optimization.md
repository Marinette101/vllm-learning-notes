# Module 4: Hardware Interaction and Kernel Co-Design

High-performance LLM serving engines achieve peak efficiency through tight co-design between software algorithms and hardware micro-architecture. While high-level scheduling algorithms govern request lifecycles, execution speed ultimately depends on how efficiently CUDA/HIP kernels utilize **GPU Memory Hierarchy**, **SRAM Tiling**, **PagedAttention Kernel Mechanics (V1 vs. V2)**, and **Cross-Hardware Accelerator Backends**.

This module explores the hardware engineering principles underlying vLLM's custom kernel stack across NVIDIA GPUs, AMD ROCm accelerators, Google TPUs, and AWS Neuron hardware.

---

## Part 1: GPU Memory Hierarchy and Bandwidth Realities

To understand kernel optimization, hardware engineers model the full accelerator node as a multi-level storage pyramid. Memory locations differ drastically in **storage capacity**, **transfer bandwidth**, and **access latency**, extending from ultra-fast on-chip GPU registers all the way down to host CPU system RAM and NVMe SSD storage.

```mermaid
flowchart TD
    REG["⚡ Registers (Per-Thread)<br>Capacity: ~65,536 x 32-bit per SM | Latency: 0 Cycles | Bandwidth: Ultra-Fast"] --> SRAM["🔥 Shared Memory / L1 Cache (On-Chip SRAM)<br>Capacity: 228 KB per SM | Latency: ~20..30 Cycles | Bandwidth: ~19 to 33 TB/s"]
    SRAM --> L2["🚀 L2 Cache (On-Chip Global)<br>Capacity: 50 MB to 256 MB | Latency: ~150..200 Cycles | Bandwidth: ~6 to 12 TB/s"]
    L2 --> HBM["🐢 High-Bandwidth Memory (Off-Chip HBM3 / HBM3e)<br>Capacity: 80 GB to 141 GB | Latency: ~400..800 Cycles | Bandwidth: 3.35 to 4.8 TB/s"]
    HBM -->|"PCIe 5.0 x16 / NVLink-C2C Bus"| CPU_RAM["💻 Host CPU System RAM (DDR5)<br>Capacity: 512 GB to 2 TB | Latency: ~100 ns | Bandwidth: ~64 GB/s (PCIe) / ~900 GB/s (NVLink-C2C)"]
    CPU_RAM -->|"PCIe Gen5 NVMe Controller"| NVME["💾 Host NVMe SSD Flash (NAND)<br>Capacity: 1 TB to 30 TB | Latency: ~10..100 us | Bandwidth: 7 to 14 GB/s"]
```

### 1.1 The Complete Accelerator Storage Pyramid

On a modern enterprise LLM node (e.g., NVIDIA H100 SXM5 server node):

| Memory Tier | Physical Location | Capacity (H100 Node) | Peak Bandwidth | Access Latency | Primary Serving Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Registers** | On-Chip (SM) | $256 \text{ KB}$ per SM | $> 100 \text{ TB/s}$ | $0 \text{ cycles}$ | Active thread scalar math & accumulators |
| **Shared Memory (SRAM)** | On-Chip (SM) | $228 \text{ KB}$ per SM | $\sim 33 \text{ TB/s}$ | $\sim 20 \dots 30 \text{ cycles}$ | SRAM tiling for $Q, K, V$ attention blocks |
| **L2 Cache** | On-Chip (Global) | $50 \text{ MB}$ | $\sim 12 \text{ TB/s}$ | $\sim 150 \dots 200 \text{ cycles}$ | Caching frequently accessed metadata & Block Tables |
| **HBM3 Memory** | Off-Chip (DRAM) | $80 \text{ GB}$ | $3.35 \text{ TB/s}$ | $\sim 400 \dots 800 \text{ cycles}$ | Active model weights & physical KV cache blocks |
| **CPU System RAM (DDR5)** | Host Motherboard | $512 \text{ GB} \dots 2 \text{ TB}$ | $\sim 64 \text{ GB/s}$ (PCIe 5.0) | $\sim 100 \text{ ns}$ | **vLLM CPU KV Cache Swapping** (`cpu_swap_space`) |
| **NVMe SSD Flash (NAND)** | PCIe NVMe Slot | $1 \text{ TB} \dots 30 \text{ TB}$ | $7 \dots 14 \text{ GB/s}$ | $10 \dots 100 \ \mu\text{s}$ | Cold-start weight loading & disk prefix cache |

---

### 1.2 The Memory Access Latency Penalty

When a CUDA thread block requests a data payload (e.g., a Key tile vector) that is not cached in SRAM or L2:

1. **Global HBM Request**: The Streaming Multiprocessor (SM) issues a global memory fetch command over the HBM memory bus.
2. **Stall Cycles**: The warp must wait $\sim 400 \dots 800 \text{ clock cycles}$ for data to travel from off-chip HBM into registers.
3. **Warp Scheduler Context Switching**: To prevent SM hardware idle time, the hardware warp scheduler context-switches to an alternate active warp. However, if all active warps are stalled waiting for HBM memory transfers, **the SM stalls completely**.

Kernel optimization minimizes HBM memory requests by staging matrix tiles inside **On-Chip Shared Memory (SRAM)** and maximizing tile reuse.

---

### 1.3 Beyond HBM: Host CPU RAM and NVMe SSD Storage Tiers

While high-speed computation takes place entirely inside GPU HBM and SRAM, vLLM utilizes the lower storage tiers (Host CPU System RAM and NVMe SSD Flash) for specialized memory management:

1. **Host CPU System RAM (DDR5) and KV Cache Swapping**:
   When GPU HBM is fully saturated under heavy preemption, vLLM's Block Manager does not discard active request KV blocks. Instead, it evacuates physical KV blocks from GPU HBM to Host CPU RAM via asynchronous CUDA transfers (`cudaMemcpyAsync`) over the PCIe 5.0 x16 bus ($\sim 64 \text{ GB/s}$). When GPU memory frees up, the blocks are swapped back to GPU HBM.
2. **Grace Hopper / Grace Blackwell (NVLink-C2C Integration)**:
   On unified architecture chips (e.g., NVIDIA GH200 / GB200), CPU System RAM and GPU HBM are connected via **NVLink-C2C** providing **$900 \text{ GB/s}$ bidirectional bandwidth**. This reduces CPU-GPU swapping latency by $> 14\times$ compared to standard PCIe Gen5.
3. **NVMe SSD Flash (NAND) and Disk Storage**:
   NAND Flash NVMe SSDs provide massive multi-terabyte capacity at $7 \dots 14 \text{ GB/s}$. vLLM uses NVMe Flash for **cold-start model weight loading** (streaming weights into HBM during startup via `safetensors`) and **persistent disk-level prefix caching** (storing static system prompt KV blocks across server restarts).

---

## Part 2: SRAM Tiling and Attention Kernel Mechanics

In standard Self-Attention ($A = \text{Softmax}(Q @ K^T / \sqrt{d}) @ V$), naive PyTorch implementations instantiate full $S \times S$ attention score matrices in global HBM memory.

For a sequence of length $S = 32,768$:

$$
\text{Memory}_{\text{naive_attn}} = S \times S \times \text{sizeof(float32)} = 32,768 \times 32,768 \times 4 \text{ bytes} \approx \mathbf{4.29 \text{ GB per head!}}
$$

Across 64 heads and 80 layers, naive attention requires terabytes of intermediate memory, causing memory overflow ($O(S^2)$ memory footprint).

### 2.1 FlashAttention Tiling Strategy

**FlashAttention** (Dao et al.) and **PagedAttention** overcome $O(S^2)$ memory explosion by tiling Query ($Q$), Key ($K$), and Value ($V$) tensors into small blocks that fit entirely inside **On-Chip SRAM ($228 \text{ KB}$)**.

```mermaid
flowchart TD
    HBM_Q["Off-Chip HBM<br>Global Q Tensor"] -->|"Load Q Tile (B_M x d)"| SRAM_Q["On-Chip SRAM<br>Q Tile"]
    HBM_K["Off-Chip HBM<br>Global K Tensor"] -->|"Load K Tile (B_N x d)"| SRAM_K["On-Chip SRAM<br>K Tile"]
    HBM_V["Off-Chip HBM<br>Global V Tensor"] -->|"Load V Tile (B_N x d)"| SRAM_V["On-Chip SRAM<br>V Tile"]

    subgraph ON_CHIP ["🔥 SM Shared Memory SRAM Compute Loop"]
        SRAM_Q --> S_TILE["SRAM Score Matrix Tile<br>S_tile = Q_tile @ K_tile^T / sqrt(d)"]
        SRAM_K --> S_TILE
        S_TILE --> O_TILE["Online Softmax and Weighted Sum<br>O_tile = Softmax(S_tile) @ V_tile"]
        SRAM_V --> O_TILE
    end

    O_TILE -->|"Write Final Output Vector"| HBM_O["Off-Chip HBM<br>Global Output Tensor O"]
```

#### Tiling Parameters
- $B_M$: Query tile block size along sequence dimension (typically $B_M = 64$ or $128$).
- $B_N$: Key/Value tile block size along sequence dimension (typically $B_N = 64$ or $128$).
- $d$: Head dimension ($d_{\text{head}} = 128$).

By executing matrix multiplication ($S_{\text{tile}} = Q_{\text{tile}} @ K_{\text{tile}}^T$) and weighted summation ($O_{\text{tile}} = P_{\text{tile}} @ V_{\text{tile}}$) entirely inside SRAM, **the $S \times S$ attention matrix is never materialized in global HBM memory**, reducing memory footprint from $O(S^2)$ to $O(S)$.

---

### 2.2 Numerical Stability: Online Softmax Recurrence

Standard Softmax requires two full passes over the input vector: Pass 1 to find global maximum $m = \max(x_i)$, and Pass 2 to compute normalization denominator $d = \sum e^{x_i - m}$.

To evaluate Softmax incrementally tile-by-tile inside SRAM without making multiple passes over HBM, attention kernels use **Online Softmax** (Rabe & Dong).

When transitioning from tile $j-1$ to tile $j$:

$$
m^{(j)} = \max\left(m^{(j-1)}, \tilde{m}^{(j)}\right)
$$

$$
l^{(j)} = l^{(j-1)} \cdot e^{m^{(j-1)} - m^{(j)}} + \tilde{l}^{(j)} \cdot e^{\tilde{m}^{(j)} - m^{(j)}}
$$

$$
O^{(j)} = O^{(j-1)} \cdot \left(\frac{l^{(j-1)} \cdot e^{m^{(j-1)} - m^{(j)}}}{l^{(j)}}\right) + \tilde{O}^{(j)} \cdot \left(\frac{e^{\tilde{m}^{(j)} - m^{(j)}}}{l^{(j)}}\right)
$$

Where:
- $m^{(j)}$: Running maximum logit score through tile $j$.
- $l^{(j)}$: Running sum of unnormalized exponentials through tile $j$.
- $O^{(j)}$: Running unnormalized attention output vector through tile $j$.

Online Softmax updates the running output vector $O^{(j)}$ in SRAM using a single forward pass over KV tiles!

---

## Part 3: PagedAttention CUDA Kernel Engineering (V1 vs. V2)

While FlashAttention assumes contiguous $K$ and $V$ tensors in HBM, **PagedAttention** accesses physical Key and Value blocks scattered across non-contiguous HBM addresses via `Block_Table` lookups.

### 3.1 PagedAttention V1 Kernel Architecture

In **PagedAttention V1** (`paged_attention_v1_kernel`), the CUDA kernel assigns **one thread block per query head per sequence**.

```mermaid
flowchart TD
    GRID["CUDA Execution Grid<br>gridDim = (num_heads, num_sequences)"] --> TB0["Thread Block (head=0, seq=0)"]
    
    subgraph V1_LOOP ["Sequential Loop inside Thread Block"]
        TB0 --> B0["Read Block_Table[0] -> Physical Block 104<br>Fetch K, V Tiles -> Compute SRAM Math"]
        B0 --> B1["Read Block_Table[1] -> Physical Block 28<br>Fetch K, V Tiles -> Compute SRAM Math"]
        B1 --> BN["... Loop sequentially through all B Physical Blocks!"]
    end

    BN --> OUT["Write Final Output O to HBM"]
```

#### Limitations of PagedAttention V1
During autoregressive decoding at long context lengths (e.g., $S = 32,768$ tokens $\equiv 2,048$ physical blocks):

- Single-request decoding has batch size $b = 1$ or small batch sizes.
- A single thread block must loop sequentially through all $2,048$ physical blocks.
- **SM Wave Under-utilization**: The GPU grid contains very few thread blocks ($\text{gridDim} = h_q \times b = 64 \times 1 = 64 \text{ thread blocks}$). On an NVIDIA H100 SXM5 with 132 SMs, **over half of the GPU's hardware SMs sit completely idle** while a few active SMs loop sequentially through thousands of blocks!

---

### 3.2 PagedAttention V2 Architecture and Split-KV Reduction

To eliminate SM wave under-utilization during long-context decoding, vLLM engineered **PagedAttention V2 (`paged_attention_v2_kernel` + Split-KV Reduction)**.

PagedAttention V2 parallelizes attention evaluation *along the sequence length dimension* by partitioning physical blocks across multiple parallel thread blocks.

```mermaid
flowchart TD
    subgraph PHASE1 ["Phase 1: Parallel Block Partitioning (paged_attention_v2_kernel)"]
        TB1["Thread Block Partition 0<br>(Blocks 0..255)"] -->|"Compute Partial Math"| TMP1["Write to Global Workspace Buffer<br>tmp_out[0], tmp_max[0], tmp_exp[0]"]
        TB2["Thread Block Partition 1<br>(Blocks 256..511)"] -->|"Compute Partial Math"| TMP2["Write to Global Workspace Buffer<br>tmp_out[1], tmp_max[1], tmp_exp[1]"]
        TB3["Thread Block Partition P...<br>(Blocks ... 2047)"] -->|"Compute Partial Math"| TMP3["Write to Global Workspace Buffer<br>tmp_out[P], tmp_max[P], tmp_exp[P]"]
    end

    subgraph PHASE2 ["Phase 2: Split-KV Reduction Kernel (paged_attention_v2_reduce_kernel)"]
        TMP1 ==> REDUCE["Launch Reduction Kernel<br>Read P Partitions from Workspace Buffer"]
        TMP2 ==> REDUCE
        TMP3 ==> REDUCE
        REDUCE --> RESCALE["Rescale and Fuse Partial Outputs via Online Softmax Identity"]
        RESCALE --> FINAL_O["Write Normalized Output Vector O to HBM"]
    end
```

#### Step 1: Parallel Block Partitioning
The scheduler sets a partition size parameter `partition_size` (typically 256 or 512 tokens per partition).

$$
N_{\text{partitions}} = \left\lceil \frac{\text{seq_len}}{\text{partition_size}} \right\rceil
$$

The CUDA grid size expands along the partition dimension ($\text{gridDim} = h_q \times b \times N_{\text{partitions}}$). For a 32K context sequence, $N_{\text{partitions}} = 32,768 / 256 = 128$ parallel thread blocks.

Instead of 64 thread blocks, the grid launches $64 \times 128 = 8,192$ parallel thread blocks, **fully saturating all 132 SMs on the GPU**!

#### Step 2: Global Workspace Intermediate Staging
Each parallel partition thread block evaluates its assigned subset of physical blocks and writes three partial arrays to a temporary global HBM workspace buffer:

1. `tmp_output[partition_idx]`: Unnormalized partial attention output vector $O_{\text{part}}$.
2. `tmp_max_logits[partition_idx]`: Maximum logit score $m_{\text{part}}$ within partition.
3. `tmp_exp_sums[partition_idx]`: Unnormalized exponential sum $l_{\text{part}}$ within partition.

#### Step 3: Split-KV Reduction Kernel (`paged_attention_v2_reduce_kernel`)
Immediately following Phase 1, a lightweight reduction CUDA kernel launches.

For each sequence head, the reduction kernel reads the $N_{\text{partitions}}$ partial vectors from the global workspace buffer, finds the global maximum logit $m_{\text{global}} = \max(m_{\text{part}, p})$, rescales partial outputs using the Online Softmax identity:

$$
\text{Scale}_p = e^{m_{\text{part}, p} - m_{\text{global}}}
$$

$$
l_{\text{global}} = \sum_{p=1}^{N_{\text{partitions}}} l_{\text{part}, p} \cdot \text{Scale}_p
$$

$$
O_{\text{final}} = \frac{\sum_{p=1}^{N_{\text{partitions}}} O_{\text{part}, p} \cdot l_{\text{part}, p} \cdot \text{Scale}_p}{l_{\text{global}}}
$$

And writes the exact, normalized attention vector $O_{\text{final}}$ to global memory.

#### Performance Impact
PagedAttention V2 increases long-context decoding speed by **$2.0\times \dots 5.0\times$** by converting sequential block loops into massive parallel SM execution!

---

## Part 4: Advanced Accelerator Backends and Cross-Hardware Engines

While CUDA powers NVIDIA GPUs, vLLM features a modular backend architecture supporting **AMD ROCm (HIP)**, **Google TPUs**, and **AWS Neuron**.

```mermaid
flowchart TD
    CORE["vLLM Core Architecture and Scheduler"] --> C_BE["NVIDIA CUDA Backend (CUDA C++ / FlashAttention / FlashInfer)"]
    CORE --> R_BE["AMD ROCm HIP Backend (HIP C++ / Composable Kernel)"]
    CORE --> T_BE["Google TPU Backend (XLA / Paged KV Custom Calls)"]
    CORE --> N_BE["AWS Neuron Backend (Neuron Core / NKI Kernels)"]
```

---

### 4.1 AMD ROCm (HIP) Execution Engine

AMD Instinct GPUs (MI250, MI300X) execute code via the **AMD ROCm** open software stack and **HIP** programming language.

#### Micro-Architecture Differences: Wavefronts vs. Warps
- **NVIDIA CUDA**: Executes threads in groups of 32 called **Warps**.
- **AMD ROCm (CDNA Architecture)**: Executes threads in groups of 64 called **Wavefronts** (`wave64`).

#### Kernel Adaptations in vLLM ROCm (`vllm._C` / HIP)
1. **Wavefront Alignment**: Shared memory reduction trees and warp shuffle instructions (`__shfl_xor_sync`) inside PagedAttention are rewritten for 64-thread wavefront execution using HIP intrinsics (`__shfl_xor`).
2. **AMD Composable Kernel (CK)**: vLLM integrates AMD's Composable Kernel library to achieve optimized GEMM matrix multiplication on MI300X **Matrix Core Units**, matching FP16 and FP8 execution throughput.
3. **MI300X Memory Bandwidth Advantage**: With $192 \text{ GB}$ of HBM3 memory providing **$5.3 \text{ TB/s}$ memory bandwidth**, MI300X handles larger KV cache block pools directly in global memory.

---

### 4.2 Google TPU Acceleration Backend (XLA)

Google Tensor Processing Units (TPUs v4, v5e, v5p, Trillium) operate on a fundamentally different execution model governed by the **XLA (Accelerated Linear Algebra) compiler**.

#### Architecture Differences: Systolic Arrays vs. SIMT Cores
- **GPUs (SIMT)**: Compute matrix multiplication using thousands of independent thread cores.
- **TPUs (Systolic Array / MXU)**: Compute matrix multiplication using specialized **Matrix Execution Units (MXU)** that stream activations through static 2D grid ALU arrays.

#### PagedAttention Adaptation in XLA
Because XLA requires static tensor shapes at compile time:

1. **Paged KV Custom Calls**: vLLM integrates with Google's XLA PagedAttention custom calls (`vllm-tpu` backend).
2. **Static Block Table Padding**: Block tables and physical KV blocks are padded to fixed TPU memory strides (typically block size $16$ or $32$).
3. **XLA Graph Compilation**: Transformer layer execution is compiled into unified XLA HLO graphs, eliminating host overhead and maximizing MXU matrix multiplication throughput.

---

### 4.3 AWS Neuron Engine (Inferentia2 & Trainium)

AWS Inferentia2 (`inf2`) and Trainium (`trn1`) instances utilize custom **Neuron Cores** managed by the `aws-neuron-sdk`.

#### Neuron Kernel Integration (NKI)
1. **Neuron Core Architecture**: Each Neuron Core contains a Tensor Engine (matrix math), Vector Engine, and Scalar Engine linked to $32 \text{ GB}$ of high-bandwidth memory.
2. **Neuron Kernel Interface (NKI)**: vLLM leverages AWS NKI to write custom PagedAttention block lookup kernels that execute directly on Neuron Core Vector and Tensor engines.
3. **Collective Communication**: Inter-core communication for Tensor Parallelism is executed across dedicated **NeuronLink** ring interconnects.

---

## Summary: Hardware Backend Capabilities Matrix

| Hardware Platform | Primary Architecture | SIMD Unit | Peak Memory Bandwidth | Primary PagedAttention Implementation |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA Hopper (H100/H200)** | GPU (Hopper / Ada) | Warp (32 Threads) | $3.35 \dots 4.8 \text{ TB/s}$ | PagedAttention V2 (CUDA C++) / FlashInfer |
| **AMD Instinct (MI300X)** | GPU (CDNA 3) | Wavefront (64 Threads) | **$5.3 \text{ TB/s}$** | PagedAttention (HIP C++) / Composable Kernel |
| **Google TPU (v5p / Trillium)** | Systolic Array (MXU) | Vector / MXU Tile | High HBM | XLA Paged KV Custom Calls |
| **AWS Trainium / Inferentia2** | Neuron Core | Tensor / Vector Engine | $820 \text{ GB/s}$ | AWS NKI Custom PagedAttention Kernels |

With **Module 4** complete, we transition to **Module 5: Distributed Parallelism (`05_distributed_parallelism.md`)**, where we explore how vLLM partitions massive frontier models across multiple GPUs and multi-node clusters using **Tensor Parallelism (Megatron-LM)**, **Pipeline Parallelism**, **Context Parallelism (Ring-Attention)**, and **Expert Parallelism (MoE Routing)**.
