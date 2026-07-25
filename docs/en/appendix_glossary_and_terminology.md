# Appendix: Master Glossary and Terminology Reference

This authoritative reference document compiles all mathematical, architectural, hardware, and distributed systems terminology utilized across **vLLM Core Architecture (`Modules 1 through 6`)**. Each entry provides exact physical definitions, mathematical formulas, and systems engineering roles.

---

## 1. Architectural and Mathematical Terminology

### `RoPE` (Rotary Position Embedding)
- **Definition**: A positional encoding mechanism (introduced by Su et al., 2021) that injects token order information into Transformer models by applying rotation matrices directly to the Query (`Q`) and Key (`K`) vectors prior to dot-product attention evaluation.
- **Mathematical Mechanics**:
  Instead of adding an absolute position vector $P_m$ to the token embedding ($X + P_m$), `RoPE` pairs adjacent dimensions inside the attention head vector (`head_dim = 128`) into 2D planes ($64 \text{ pairs}$). For a token located at sequence index $m$, each 2D plane is multiplied by a 2D rotation matrix $R_{m,\theta_i}$:

$$
R_{m, \theta_i} = \begin{bmatrix} \cos(m \cdot \theta_i) & -\sin(m \cdot \theta_i) \\ \sin(m \cdot \theta_i) & \cos(m \cdot \theta_i) \end{bmatrix}
$$

  When evaluating the dot-product between Query at position $m$ ($Q_m$) and Key at position $n$ ($K_n$), orthogonal rotation algebra guarantees that the resulting inner product depends **strictly on the relative distance ($m - n$) between the two tokens**:

$$
\text{Score} = (R_m @ Q_m) @ (R_n @ K_n)^T = Q_m @ R_{m-n} @ K_n^T
$$

- **Architectural Significance**: Eliminates absolute position artifacts, preserves base semantic vector magnitudes, and enables seamless long-context extrapolation (`e.g., scaling from 8K to 128K context via NTK-aware scaling and YaRN`).
- **Exact Matrix Dimensions of $R_m$ (`[128, 128]` Block-Diagonal)**:
  `RoPE` does not rotate across different attention heads (`num_kv_heads = 8`), because each head encodes an orthogonal semantic domain. Instead, `RoPE` operates strictly inside the `128` feature coordinates of each individual head (`Axis m = 0..127`), grouping them into sixty-four 2D pairs. Written as a single operator acting on a `[128]` head vector, $R_m$ is a **`[128, 128]` block-diagonal matrix containing sixty-four `2x2` rotation blocks along its diagonal**. Consequently, all `64` Query heads and `8` Key heads undergo this `[128, 128]` rotation independently (`head count acts as a parallel batch index`). Furthermore, historical keys in the `KV` cache ($k_0, k_1, \dots, k_{n-1}$) are stored already rotated ($R_m @ k_m$) and are never re-rotated during step $n$.

- **Symmetry and Determinism of $R_m$ (`Q/K Symmetry and Theta Hierarchy`)**:
    1. **`Q/K` Symmetry**: At sequence position $m$, the exact same rotation matrix $R_m$ multiplies both Query ($Q_m$) and Key ($K_m$). This symmetry is why $\text{Score} = Q_m @ K_n^T$ simplifies via orthogonal identity ($R_m^T @ R_m = I$) directly into $Q_{\text{unrotated}} @ R_{m-n} @ K_{\text{unrotated}}^T$.
    2. **Universal Head Application**: $R_m$ applies identically across every single one of the `64` Query heads and `8` Key heads.
    3. **Strict Determinism**: $R_m$ is completely independent of the token's semantic text or activation value (`"Apple"` vs `"Dog"` yields the exact same $R_m$ at position $m$). It is governed strictly by two variables: the integer sequence index $m$ (`0, 1, 2, ...`) and the geometric frequency hierarchy $\theta_i = 10000^{-2i / d_{\text{head}}}$ where $i \in [0, 63]$.
- **Chronological Execution Pipeline (`Embedding vs. RoPE`)**:
  Initial token embedding lookup ($W_{\text{Embed}}$) and `RoPE` rotation operate at two distinct execution stages:

    1. **Step 0 (`Embedding Lookup`)**: Converts raw token IDs into unrotated base tensor $X_0$ (`[b, s, d]`). Contains **zero positional information**.
    2. **Step 1 (`Linear Projections`)**: Inside every layer, $X$ multiplies against $W_Q$ and $W_K$ to produce unrotated $Q$ and $K$ tensors.
    3. **Step 2 (`RoPE Rotation`)**: Strictly **after** `Q/K` linear projections and **before** dot-product attention, `RoPE` rotation matrices ($R_m$) multiply against $Q$ and $K$ ($Q_{\text{rotated}} = R_m @ Q_{\text{unrotated}}$), ensuring dot-products evaluate pristine relative distances without projection distortion across layers.

### `SwiGLU` (Swish-Gated Linear Unit)
- **Definition**: A gated non-linear activation architecture (introduced by Noam Shazeer) replacing classic `ReLU` Feed-Forward Networks (`FFNs`) across modern frontier transformers (`Llama 3, Mistral, DeepSeek`).
- **Mathematical Mechanics**:
  Requires three projection matrices per layer (`W_Gate`, `W_Up`, `W_Down`):
  $$
\text{FFN_Output} = \left( \text{SiLU}(H @ W_{\text{Gate}}) \odot (H @ W_{\text{Up}}) \right) @ W_{\text{Down}}
$$
  Where $\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$ acts as a continuous, differentiable gating modulation curve, and `(*)` denotes element-wise Hadamard multiplication across matching coordinate indices.

### `MHA` (Multi-Head Attention)
- **Definition**: The original standard attention architecture (`Vaswani et al., 2017`) where every Query head (`h_q`) is paired with an independent, dedicated Key (`h_k`) and Value (`h_v`) projection head (`h_q : h_kv = 1 : 1`).
- **Trade-off**: Maximizes representation capacity but incurs severe memory footprint scaling during long-context autoregressive decoding due to massive `KV` cache retention.

### `MQA` (Multi-Query Attention)
- **Definition**: An attention variant (`Shazeer, 2019`) where all `num_q_heads` (`e.g., 64`) share **one single Key and Value head (`h_kv = 1`)**.
- **Trade-off**: Drastically compresses `KV` cache footprint (`h_q : 1` reduction) but induces representation collision and capacity saturation, causing quality degradation on complex multi-hop mathematical reasoning.

### `GQA` (Grouped-Query Attention)
- **Definition**: The Pareto-optimal attention architecture (`Ainslie et al., 2023`) partitioning Query heads into distinct local groups that share a single Key/Value head (`e.g., Llama 3 70B uses 8 KV heads for 64 Q heads, an 8:1 ratio`).
- **Trade-off**: Balances near-`MHA` evaluation quality against an `8x` reduction in `KV` cache memory footprint (`HBM`).

### `MLA` (Multi-Head Latent Attention)
- **Definition**: A low-rank joint `KV` compression architecture (`DeepSeek-V2/V3/R1`) designed to compress memory footprint across ultra-large frontier models without sacrificing reasoning fidelity.
- **Mathematical Mechanics**:
  Down-projects input activation `H` into a single low-rank latent vector `c_kv` (`d_c = 512`), which serves as the sole cached content payload. To prevent positional rotation entanglement, `RoPE` is decoupled into a dedicated 64-dimensional positional key vector (`k_pe`):
  $$
\text{Total Cached Vector per Token} = c_{kv} \ (512 \text{ dims}) + k_{pe} \ (64 \text{ dims}) = \mathbf{576 \text{ floating-point numbers}}
$$
  During attention evaluation, full `K_content` and `V` vectors are restored on-the-fly inside fast GPU `SRAM` (`c_kv @ W_UK` and `c_kv @ W_UV`) or absorbed directly into Query weight matrices (`Matrix Absorption`).

### `MoE` (Mixture of Experts)
- **Definition**: A sparse neural network architecture decoupling total parameter capacity from active inference compute (`FLOPs`) by replacing dense `FFN` layers with multiple parallel expert `FFNs` governed by a dynamic Router (`Gating Network`).
- **Mechanics**: For each token, the Router evaluates similarity logits against all experts (`H @ W_Router`), selects the `Top-K` highest logits (`e.g., Top-2 in Mixtral 8x7B`), and applies `Softmax` over the selected subset to weight and synthesize the expert outputs.

---

## 2. Linear Algebra and Execution Operations

### `GEMM` (General Matrix-Matrix Multiplication)
- **Definition**: The primary linear algebra operation governing compute-bound neural network layers (`C = A @ B + D`).
- **Inference Context**: Dominates the **Prefill Phase**, where all prompt tokens (`s_prompt > 1`) are processed simultaneously (`[batch_size * seq_len, hidden_dim] @ [hidden_dim, out_dim]`), saturating GPU Tensor Cores at high Arithmetic Intensity.

### `GEMV` (General Matrix-Vector Multiplication)
- **Definition**: The linear algebra operation where a matrix multiplies against a single vector (`c = A @ b`).
- **Inference Context**: Dominates the **Decode Phase** (`batch_size = 1`), where only a single new token (`seq_len = 1`) is evaluated per step (`[1, hidden_dim] @ [hidden_dim, out_dim]`). `GEMV` operates deep inside the memory-bandwidth-bound regime (`I_decode =~ 1.0 to 2.0 FLOPs/Byte`).

### `BMM` (Batched Matrix Multiplication)
- **Definition**: A parallel matrix multiplication operation (`torch.matmul` or `A @ B`) across higher-dimensional tensors (`[b, h, m, k] @ [b, h, k, n] = [b, h, m, n]`).
- **Mechanics**: Operates exclusively on the last two dimensions (`[-2, -1]`), treating all preceding leading dimensions (`[:-2]`) as independent parallel batch indices dispatched across GPU Streaming Multiprocessors (`SMs`).

---

## 3. Hardware and Memory Hierarchy Terminology

### `HBM` (High-Bandwidth Memory)
- **Definition**: Off-chip DRAM stacks connected directly to the GPU silicon die via an ultra-wide silicon interposer (`e.g., 80 GB or 192 GB capacity on NVIDIA H100/H200`).
- **Role and Bandwidth**: Serves as the primary global memory pool holding model weights and static `KV` cache blocks. Delivers `3.35 TB/sec` bandwidth on H100 SXM5 (`4.5x slower than on-chip L2 cache`).

### `L2 Cache` (Level 2 Static RAM)
- **Definition**: On-chip Static Random-Access Memory (`SRAM`) physically located on the GPU silicon die, shared across all Streaming Multiprocessors (`50 MB` on H100 SXM5).
- **Role and Bandwidth**: Serves as the high-speed staging buffer (`12 to 15 TB/sec` bandwidth) between off-chip `HBM` and individual SM cores.

### `SM` (Streaming Multiprocessor) and Registers / `L1 Cache`
- **Definition**: The individual physical compute cores inside a GPU (`132 SMs on NVIDIA H100`). Each SM contains local `SRAM` (`Registers and ~256 KB L1/Shared Memory`) delivering `30 to 50+ TB/sec` bandwidth.
- **Role**: Executes square thread block tiles (`e.g., 128x128 tiles`) via specialized math execution units (`Tensor Cores`).

### `Arithmetic Intensity` (`I`) and Roofline Model
- **Definition**: The ratio of total floating-point operations performed (`FLOPs`) divided by the total physical memory bytes transferred from `HBM` (`Bytes`):
  $$
I = \frac{\text{Total_FLOPs_Performed}}{\text{Total_Bytes_Transferred_from_HBM}}
$$
- **Ridge Point (`I_ridge`)**: The exact threshold (`I_ridge = Peak_FLOPs / Peak_HBM_Bandwidth`, `~295 FLOPs/Byte on H100`) separating compute-bound execution from memory-bandwidth-bound execution.
- **Comparative Evaluation Rule**: When comparing GPUs (`A100 vs. H100 vs. MI300X`), a higher `I_ridge` does **not** indicate a better GPU for serving. Because LLM autoregressive decoding operates deep inside the memory-bound slope (`I =~ 1.0 to 2.0 FLOPs/Byte`), throughput is decided strictly by `Peak_HBM_Bandwidth (beta)`. Therefore, a **LOWER `I_ridge` combined with higher `HBM` bandwidth** (`e.g., AMD MI300X at 245 FLOPs/Byte and 5.3 TB/sec`) is far superior for decoding, whereas high `I_ridge` is advantageous only for compute-bound training and large-batch prompt prefill.
- **Canonical Batch-1 Decoding Range (`~1.0 to 2.0 FLOPs/Byte`)**: In systems engineering literature, single-batch decoding is universally cited between `1.0` and `2.0 FLOPs/Byte` (`most frequently ~1.8 to 1.9 FLOPs/Byte`). This reflects exact physical assumptions: isolated `FP16` projection math yields `1.0 FLOPs/Byte`, isolated `FP8/INT8` projection math yields `2.0 FLOPs/Byte`, and end-to-end `FP16` decoding (`incorporating both weight transfers and dynamic KV cache/attention dot-product FLOPs across sequence length s`) converges tightly between `~1.8 and 1.9 FLOPs/Byte`.

---

## 4. Distributed Systems and Parallelism Terminology

### `EP` (Expert Parallelism)
- **Definition**: A distributed partitioning strategy for `MoE` architectures where different physical GPU nodes or devices store disjoint subsets of expert weight tables (`e.g., GPU 0 stores Expert 0, GPU 7 stores Expert 7`).
- **Execution Mechanics**: Requires `All-to-All` collective communication across high-speed interconnects (`NVLink` or `RDMA`) to dispatch token activation vectors (`H`) to their designated expert GPUs and return the computed transformations.

### `TP` (Tensor Parallelism)
- **Definition**: An intra-node distributed partitioning strategy (`Megatron-LM`) slicing individual linear weight matrices (`W_Q, W_K, W_V, W_1, W_2`) across multiple physical GPUs within the same server node.
- **Execution Mechanics**: Executes column-parallel or row-parallel matrix transformations, requiring sub-millisecond `AllReduce` synchronization across `NVLink / NVSwitch` after every transformer layer.

### `PP` (Pipeline Parallelism)
- **Definition**: An inter-node distributed partitioning strategy dividing consecutive transformer layers across sequential physical server nodes (`e.g., Node 0 runs Layers 1-20, Node 1 runs Layers 21-40`).
- **Execution Mechanics**: Utilizes micro-batching (`1F1B schedule`) to overlap compute across stages while mitigating pipeline bubble latency.

---

## 5. Serving Engine and Paged Memory Terminology

### `PagedAttention`
- **Definition**: The foundational attention kernel of vLLM adapted from operating system virtual memory paging. It decouples logical token sequence order from physical contiguous `HBM` locations, enabling attention scores to be evaluated across non-contiguous `16-token` physical blocks in $\mathcal{O}(1)$ lookup time without memory fragmentation.

### `CoW` (Copy-on-Write)
- **Definition**: A block management mechanism inside vLLM where multiple logical requests (`e.g., shared system prompts, beam search branches, or parallel best_of_n sampling`) point to the exact same physical `KV` cache blocks with reference counting (`ref_count > 1`). Physical duplication occurs strictly when a specific request diverges during generation.

### `APC` (Automatic Prefix Caching)
- **Definition**: A hash-based block management algorithm inside vLLM that calculates cryptographic hashes across token sequences to identify and retain reusable prefix blocks across independent, asynchronous user requests.

### `Block Table`
- **Definition**: A per-sequence metadata lookup table in vLLM that translates logical block indices (`e.g., Logical Block 1`) to physical `HBM` block pointers (`e.g., Physical Block 28`) in $\mathcal{O}(1)$ time. Also tracks `num_filled_tokens` and block reference counts (`ref_count`).

### `Online Softmax`
- **Definition**: The mathematical algorithm (`Milakov and Gimelshein, 2018`) utilized inside PagedAttention kernels that maintains running maximum `m_i` and running exponent sum `l_i` across block iterations, allowing exact `Softmax` and Value accumulation to be computed in on-chip `SRAM` registers across non-contiguous physical blocks without intermediate `HBM` round-trips.

### `Continuous Batching` (`Iteration-Level Scheduling`)
- **Definition**: A dynamic scheduling paradigm (`Orca, Yu et al., 2022`) where the inference server inspects the request queue on every single token generation tick (`iteration`), ejecting completed sequences (`EOS`) and admitting new requests (`Waiting -> Running`) immediately, eliminating the up to 90% GPU idle waste of static request-level batching.

### `Preemption` (`Swap vs. Recomputation`)
- **Definition**: The recovery mechanism triggered when the `HBM` free block pool is exhausted during autoregressive decoding. Low-priority `Running` sequences are demoted to `Swapped` (`evicting physical KV blocks over PCIe to host CPU RAM`) or `Waiting` (`discarding KV blocks entirely for later recomputation`), freeing physical blocks for active generation.

### `Internal Fragmentation`
- **Systems Engineering Definition**: Memory waste occurring **INSIDE** the boundary of an allocated physical memory region because the allocated block size exceeds the payload currently stored within it.
- **LLM Serving Impact**: Under contiguous pre-allocation (`max_seq_len`), unused reserved slots inside a sequence's allocated chunk remain locked, wasting 60%-80% of `HBM`. In vLLM, internal fragmentation is eliminated for all full physical blocks and isolated strictly to the active tail block (`< 4%` average waste).

### `External Fragmentation`
- **Systems Engineering Definition**: Memory waste occurring **OUTSIDE** of all allocated memory regions when total unallocated free memory exists across physical address space, but is fragmented into non-contiguous gaps, causing contiguous allocation requests to fail with Out-Of-Memory (`OOM`).
- **LLM Serving Impact**: Legacy serving engines suffer 5%-10% external fragmentation gridlock due to physical contiguity rules. vLLM reduces external fragmentation to **exactly 0%** by evaluating attention across non-contiguous fixed-size blocks via PagedAttention.
