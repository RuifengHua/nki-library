# Attention TKG Sharding Modes Design Document

## Overview

This document describes the design and implementation of KV cache sharding strategies in the attention_block_tkg kernel, focusing on KV data parallelism.

For flash attention (tiled KV processing for long context), see [attention_tkg_design_spec.md](attention_tkg_design_spec.md).

## Definitions

    Term        Definition                                                              GPT-OSS 120B (TP64)
    ----        ----------                                                              -------------------
    B           Batch size                                                              1-512
    q_heads     Number of Q heads per rank                                              1 (TP64)
    kv_heads    Number of KV heads per rank                                             1
    s_prior     Prior context length, also called S_ctx                                 up to 128K
    s_active    Active sequence length (tokens being generated), also called S_tkg      1-8
    d_head      Attention head dimension                                                64
    n_layers    Number of transformer layers                                            80
    block_len   Number of tokens per KV cache block                                     16-256
    TP          Tensor Parallelism - shard on attention heads                           64
    KVDP        KV data parallelism - shard KV cache on batch dimension                 8
                (also called batch sharding)
    CP          Context parallelism - shard KV cache on s_prior dimension               8
                (also called flash decode)
    DP          Data parallelism - shard KV cache on batch, replicate Q/out projections 8
                (also called DP attention)
    GQA         Grouped Query Attention - multiple Q heads share one KV head            8 Q heads per KV head
    LNC2        Logical NeuronCore 2 - two physical NCs acting as one logical NC        -
    SBUF        State Buffer - on-chip SRAM for intermediate data                       -
    HBM         High Bandwidth Memory - off-chip DRAM                                   -

## Problem Statement: KV Cache Sharding

KV cache size scales linearly with context (s_prior) and batch. 
For long context inference (s_prior up to 1M tokens) or high throughput (large batch), 
the KV cache becomes too big to fit in HBM (24GB on Trn2, 36GB on Trn3).

Sharding types:
    K/V cache shape: [n_layers, batch, kv_heads, s_prior, d_head]
                                ^^^^^  ^^^^^^^^  ^^^^^^^
                               KV data  vanilla  context
                               parallel  (TP)    parallel
                                (KVDP)            (CP)

In vanilla TP, we shard on attention heads (q_heads) and each rank in a grouped query attention (GQA) group maintains a full copy of the KV cache. 
For example, TP64 on GPT-OSS 120B (64 Q heads and 8 KV heads) will have 1 Q head and 1 KV head per core. Resulting in 8X KV cache replication.
We can eliminate this replication by using TP = KV heads, so TP8 for attention (1 KV head and 8 Q heads per core). 

The remaining 8x parallelism can be used to shard the KV cache:
    TP8 KVDP8 (KV Data Parallel): Shard KV cache along batch dimension. Each rank handles B/8 batches.
    TP8 CP8 (Context Parallelism): Shard KV cache along s_prior dimension. Each rank handles s_prior/8 context length.
Both reduce the KV cache by the same amount.

Sharding Recommendations:
    Use vanilla TP = kv_heads
    B >= global_ranks/kv_heads: Use KV data parallelism
    B < global_ranks/kv_heads: Use context parallelism or combine KV data parallelism with context parallelism

For example, assuming GQA: q_heads=64, kv_heads=8, global_ranks=64
    B=1 Use TP8-CP8
    B=2 Use TP8-CP8 or TP8-KVDP2-CP4
    B=4 Use TP8-CP8 or TP8-KVDP4-CP2
    B=8 and higher: Use TP8-KVDP8

## Out of Scope: Data Parallel (DP) Attention

Both KV data parallelism and DP attention shard KV cache along the batch dimension (each rank has B/KVDP batches). The difference:
- DP Attention (TP8 DP8): 
    Replicates all projection weights (W_q, W_kv, W_out) across DP groups. 
    Each of the 8 DP groups project the TP8 sharded Q heads for B/8 batches. 
    No extra communication across DP ranks is required for the Q projection since each DP group has the full Q projection for its assigned batches.
- KV data parallelism (TP64 KVDP8): 
    Uses standard TP64 for projections (W_kv replicated due to GQA). 
    Each individual rank projects the TP64 sharded Q heads for all batches. 
    In KVDP8 the prior KV is stored batch sharded within the GQA group. 
    Since each KV head within a GQA group must attend to all Q heads within its group, the TP64 sharded Q must be gathered across KVDP ranks before attention.

While DP attention uses larger weights (8x), since the X input activation is batch sharded (8x smaller), the expected number of matrix multiplications is the same across techniques.

KV data parallelism avoids additional weight replication at the cost of two extra collectives: all_gather on Q heads before attention, and all_gather on batch after attention.
This document covers KV data parallelism.
DP attention is out of scope.

### Projection Weight and KV Cache Comparison (GPT-OSS 120B: 64 Q heads, 8 KV heads)

|                     | Vanilla TP64 | KVDP (TP64 KVDP8) | CP (TP64 CP8)        | DP Attention (TP8 DP8) |
|---------------------|--------------|-------------------|----------------------|------------------------|
| W_kv replication    | 8x (GQA)     | 8x (GQA)          | 8x (GQA)             | 8x (DP)                |
| W_q replication     | 1x           | 1x                | 1x                   | 8x (DP)                |
| W_out replication   | 1x           | 1x                | 1x                   | 8x (DP)                |
| X input activation  | [B, S, H]    | [B, S, H]         | [B, S, H]            | [B/8, S, H]            |
| KV cache per rank   | B batches    | B/8 batches       | B batches, s_prior/8 | B/8 batches            |

## Block KV Cache Support

vLLM's PagedAttention divides KV cache into fixed-size blocks that can be allocated and freed independently, 
reducing fragmentation and enabling automatic prefix caching (APC). 

Performance benefits:
- Prefill: Skip computation for cached prefixes (e.g., system prompts), improving TTFT
- Decode: Skip DMA for sequences shorter than the bucket S_ctx size via `oob_mode=skip`, improving OTPS

Block KV Layout:
    Cache shape:  [num_blocks, block_len, d_head]  # pool of blocks
    Active table: [B, num_active_blocks]           # block indices per batch (padded with -1)

Example (B=8, s_prior=131072, block_len=32):

    Contiguous: K_cache[B=8, kv_heads=1, s_prior=131072, d_head=64]
    
    Block KV:
        K_cache[num_blocks=32768, block_len=32, d_head=64]  # pool of blocks
        active_blocks_table[B=8, num_active_blocks=4096]:   # 131072/32 = 4096 blocks per batch
            batch 0: [0, 8, 16, ..., -1, -1]
            batch 1: [1, 9, 17, ..., -1, -1]
            ...

    The active_blocks_table contains block indices into K_cache's first dimension.
    E.g., the second block of batch 1 is at K_cache[9, :, :].

Loading Block KV:
The attention kernel uses `dma_copy` with `vector_offset` (indirect DMA) to load
blocks using indices from `active_blocks_table`. This loads up to 128 blocks at
once into SBUF (one block per partition). Blocks with invalid indices (-1) are
skipped via `oob_mode=skip`.
The K cache is then transposed in SBUF:
    [128blks, block_len * d_head] -> [d_head, block_len * 128blks]
So that d_head on the partition dimension for the Q @ K^T matmul.

### Block KV with KV data parallelism

With KV data parallelism, each KVDP rank computes attention for B/KVDP batches. 
vLLM treats each DP rank as an independent inference endpoint with its own KV cache:
- block pool: K_cache[num_blocks_local, block_len, d_head]
- Its own block table: `active_blocks_table[B/KVDP, num_active_blocks]` with indices local to that rank's cache

Continuing the example above with block KV (B=8, s_prior=131072, block_len=32, KVDP=4):

    Rank 0: K_cache_0[8192, 32, 64], active_blocks_table_0[2, 4096]  -> batches 0,1
    Rank 1: K_cache_1[8192, 32, 64], active_blocks_table_1[2, 4096]  -> batches 2,3
    Rank 2: K_cache_2[8192, 32, 64], active_blocks_table_2[2, 4096]  -> batches 4,5
    Rank 3: K_cache_3[8192, 32, 64], active_blocks_table_3[2, 4096]  -> batches 6,7

## KV data parallelism Implementation

### Purpose

KV data parallelism reduces KV cache memory by distributing cache along the batch dimension across KVDP ranks. 
Each rank stores B/KVDP batches of KV cache instead of B.

    K/V cache shape: [n_layers, batch/KVDP, kv_heads=1, s_prior, d_head]
                                ^^^^^^^^^^^
                                KV data parallelism

### Data Flow

The attention_block_tkg kernel with KV data parallelism wraps the standard attention flow with collective operations:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                        ATTENTION BLOCK TKG WITH KV DATA PARALLELISM                            │
│                                                                                                │
│  Per-rank input: X [B, S_tkg, H]     KV cache: [B/KVDP, S_ctx, d] (pre-sharded)                │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐         │
│  │  RMSNorm X      │-->│ QKV Projection  │-->│ Split QKV->Q,K  │-->│ RMSNorm Q/K     │-->      │
│  │  (optional)     │   │                 │   │  (transpose)    │   │ (optional)      │         │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘   └─────────────────┘         │
│                                                                                                │
│      ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐                           │
│   -->│ RoPE Embedding  │-->│ RMSNorm Q/K     │-->│ Quantize K/V    │                           │
│      │ (optional)      │   │ (optional)      │   │ to FP8 (opt.)   │                           │
│      └─────────────────┘   └─────────────────┘   └────────┬────────┘                           │
│                                                           v                                    │
│  ╔════════════════════════════════════════════════════════════════════════════════════════╗    │
│  ║                       KV DATA PARALLELISM INPUT COLLECTIVES                            ║    │
│  ║  ┌─────────────────────────────────────────────────────────────────────────────────┐   ║    │
│  ║  │ 1. Rearrange Q: (q_heads, B, S, d) -> (KVDP*q_heads, B/KVDP, S, d)              │   ║    │
│  ║  │ 2. all_to_all Q dim=0: (KVDP*q_heads, B/KVDP, S, d) — all heads, local batch    │   ║    │
│  ║  │ 3. Transpose back to SBUF: (d, B_attn*q_heads_attn*S)                           │   ║    │
│  ║  │ K: Slice batch to SBUF: (d, B/KVDP*S)                                           │   ║    │
│  ║  │ V: Slice batch in HBM: (B/KVDP, 1, S, d)                                        │   ║    │
│  ║  └─────────────────────────────────────────────────────────────────────────────────┘   ║    │
│  ╚════════════════════════════════════════════════════════════════════════════════════════╝    │
│                                                       │                                        │
│                                                       v                                        │
│                   ┌──────────────────────────────────────────────────────────┐                 │
│                   │                    ATTENTION TKG                         │                 │
│                   │  Q: (d, B_attn*q_heads_attn*S) @ SBUF                    │                 │
│                   │  K: (d, B_attn*S) @ SBUF    V: (B_attn, 1, S, d) @ HBM   │                 │
│                   │  softmax(Q @ K^T / sqrt(d)) @ V                          │                 │
│                   │  Output: (d, B_attn*q_heads_attn*S) @ SBUF               │                 │
│                   └───────────────────────────────────┬──────────────────────┘                 │
│                                                       │                                        │
│                                                       v                                        │
│  ╔════════════════════════════════════════════════════════════════════════════════════════╗    │
│  ║                       KV DATA PARALLELISM OUTPUT COLLECTIVES                           ║    │
│  ║  ┌───────────────────────────────────────────────────────────────────────────────────┐ ║    │
│  ║  │ 1. Rearrange attn: (B/KVDP, KVDP*q_heads, d, S) -> (KVDP*q_heads, B/KVDP, d, S)   │ ║    │
│  ║  │ 2. all_to_all attn dim=0: (KVDP*q_heads, B/KVDP, d, S) — local heads, all batches │ ║    │
│  ║  │ 3. Rearrange back: (KVDP*q_heads, B/KVDP, d, S) -> (B, q_heads, d, S)             │ ║    │
│  ║  └───────────────────────────────────────────────────────────────────────────────────┘ ║    │
│  ╚════════════════════════════════════════════════════════════════════════════════════════╝    │
│                                                       │                                        │
│                                                       v                                        │
│                   ┌───────────────────────────────────────────────────────────┐                │
│                   │              KV Cache Update (optional)                   │                │
│                   │              Updates B/KVDP batches only                  │                │
│                   └───────────────────────────────────┬───────────────────────┘                │
│                                                       │                                        │
│                                                       v                                        │
│                   ┌───────────────────────────────────────────────────────────┐                │
│                   │              Output Projection (optional)                 │                │
│                   │              W_out @ attn -> [B, S_tkg, H]                │                │
│                   └───────────────────────────────────────────────────────────┘                │
│                                                                                                │
│  Per-rank output: [B, S_tkg, H]                                                                │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Configuration Parameters

| Parameter              | Type                | Description                                                       |
|------------------------|---------------------|-------------------------------------------------------------------|
| `KVDP`                 | int                 | KV data parallelism degree (1 = disabled)                         |
| `KVDP_replica_group`   | ReplicaGroup        | Rank group for collectives                                        |
| `KVDP_collective_mode` | KVDPCollectiveMode  | Collective mode: `ALL_TO_ALL` (default) or `ALL_GATHER_SLICE`     |
| `KVDP_rank`            | nl.ndarray          | Shape (1,), uint32 @ HBM. This rank's position in its KVDP group  |

### Collective Modes

The `KVDP_collective_mode` parameter selects the collective operation strategy for Q input and attention output redistribution:

- **ALL_TO_ALL (default):** Single `all_to_all` collective that combines gather and slice in one step. Requires Mesh algorithm (≥4 ranks lnc2 or ≥8 ranks lnc1).
- **ALL_GATHER_SLICE:** `all_gather` on heads/batch followed by a `KVDP_rank`-based slice. Works with any rank count (≥2).

Note: K/V batch slicing still uses `KVDP_rank` in both modes (local slice, not a collective).

### Collective Operations: ALL_TO_ALL

The implementation uses a single `all_to_all` collective that combines gather and slice in one step.
Rearranges are required to put KVDP batch/head groups on dim=0 for all_to_all chunking.

**Input Collectives (before attention):**

    SBUF Input               Rearrange + Transpose to HBM                  all_to_all                       Transpose + Rearrange to SBUF
    ──────────               ─────────────────────────────                  ──────────                       ─────────────────────────────
    Q (q==1): (d, B*S)       rearrange in HBM (KVDP, d, B/KVDP*S)          (KVDP, d, B/KVDP*S)              rearrange to (d, B*S) @ SBUF
    Q (q>1):  (d, B*q*S)     rearrange SBUF (d, KVDP*q, B/KVDP*S)          (KVDP*q, B/KVDP, S, d)           (d, B*q*S) @ SBUF
                             tiled transpose to HBM (KVDP*q, B/KVDP, S, d)                                  tiled transpose + rearrange
    K: (d, B*S)              slice in HBM                                  -                                (d, B/KVDP*S) @ SBUF
    V: (B, 1, S, d)          slice in HBM                                  -                                stays in HBM

When q_heads==1, the Q path avoids transpose entirely: d stays on partition dim, rearrange in HBM only.

**Output Collectives (after attention):**

    SBUF Input                 Rearrange + Transpose to HBM                  all_to_all                       Transpose + Rearrange to SBUF
    ──────────                 ─────────────────────────────                 ──────────                       ─────────────────────────────
    attn: (d, B*q*S)           rearrange SBUF (d, KVDP*q, B/KVDP*S)          (KVDP*q, B/KVDP, d, S)           (d, B*q*S) @ SBUF
                               tiled transpose to HBM (KVDP*q, B/KVDP, d, S)                                  tiled transpose + rearrange

**Example with KVDP=4, q_heads=1, B=8, S_tkg=1, d_head=64:**

    Input Collectives (B/KVDP=2, q_heads*KVDP=4):
        Q @ SBUF:  (64, 8)           - d_head=64, B*q_heads*S_tkg=8
        Q @ HBM:   (64, 8)           - copy to HBM (q_heads=1, no transpose)
        Rearranged:(4, 64, 2)        - (KVDP=4, d_head=64, B/KVDP*S=2) — KVDP groups on dim 0
        After a2a: (4, 64, 2)        - rank i gets chunk i from all ranks
        Q @ SBUF:  (64, 8)           - rearrange (q_attn=4, d=64, B_attn=2, S=1) → (d=64, B_attn*q_attn*S=8)

    Output Collectives:
        attn @ SBUF: (64, 8)         - d_head=64, B*q_heads*S_tkg=8
        Rearranged:  (64, 8)         - rearrange (d, B_attn, q_attn, S) → (d, q_attn, B_attn, S)
        attn @ HBM:  (4, 2, 64, 1)   - tiled transpose: (q_attn=4, B_attn=2, d=64, S=1)
        After a2a:   (4, 2, 64, 1)   - rank i gets q_heads from all KVDP ranks
        attn @ SBUF: (64, 8)         - tiled transpose + rearrange (d, KVDP, q, B_attn, S) → (d, B*q*S)

**Example with KVDP=4, q_heads=2, B=8, S_tkg=1, d_head=64 (q_heads>1 path):**

    Input Collectives (B_attn=B/KVDP=2, q_heads_attn=KVDP*q_heads=8):
        Q @ SBUF:  (64, 16)          - (d=64, B*q*S=8*2*1=16)
        Rearrange: (64, 16)          - rearrange SBUF (d, B, q, S)=(64,8,2,1) → (d, KVDP, q, B_attn, S)=(64,4,2,2,1)
        Q @ HBM:   (8, 64, 2, 1)     - tiled transpose: (KVDP*q=8, d=64, B_attn=2, S=1)
        After a2a: (8, 64, 2, 1)     - rank i gets chunk i (q=2 heads) from all 4 ranks
        Transpose: (64, 16)          - tiled transpose back: (d=64, q_attn*B_attn*S=8*2*1=16)
        Rearrange: (64, 16)          - rearrange SBUF (d, q_attn, B_attn, S)=(64,8,2,1) → (d, B_attn, q_attn, S)=(64,2,8,1)
        Q @ SBUF:  (64, 16)          - (d=64, B_attn*q_attn*S=16)

    Output Collectives:
        attn @ SBUF: (64, 16)        - (d=64, B_attn*q_attn*S=2*8*1=16)
        Rearrange:   (64, 16)        - rearrange SBUF (d, B_attn, q_attn, S)=(64,2,8,1) → (d, q_attn, B_attn, S)=(64,8,2,1)
        attn @ HBM:  (8, 2, 64, 1)   - tiled transpose: (q_attn=8, B_attn=2, d=64, S=1)
        After a2a:   (8, 2, 64, 1)   - rank i gets q=2 heads from all 4 ranks
        Transpose:   (64, 16)        - tiled transpose back: (d=64, q_attn*B_attn*S=16)
        Rearrange:   (64, 16)        - rearrange SBUF (d, KVDP, q, B_attn, S)=(64,4,2,2,1) → (d, KVDP, B_attn, q, S)=(64,4,2,2,1)
        attn @ SBUF: (64, 16)        - (d=64, B*q*S=8*2*1=16)

### Collective Operations: ALL_GATHER_SLICE

The implementation uses `all_gather` on heads + `slice` on batch. Transposes are required to move the
collective dimension to dim=0 for all_gather.

**Input Collectives (before attention):**

    SBUF Input               Transpose to HBM              all_gather                  slice batch                      Transpose to SBUF
    ──────────               ────────────────              ──────────                  ───────────                      ─────────────────
    Q: (d, B*q_heads*S)      (q_heads, B, S, d) @ HBM      (KVDP*q_heads, B, S, d)     (KVDP*q_heads, B/KVDP, S, d)     (d, B*q_heads*S) @ SBUF
    K: (d, B*S)              slice in HBM                  -                           (d, B/KVDP*S)                    (d, B/KVDP*S) @ SBUF
    V: (B, 1, S, d)          slice in HBM                  -                           (B/KVDP, 1, S, d)                stays in HBM

When q_heads==1, the Q transpose can be skipped: all_gather directly on d_head dimension, then rearrange.

**Output Collectives (after attention):**

    SBUF Input                 Transpose to HBM                 all_gather                 slice heads              Transpose to SBUF
    ──────────                 ────────────────                 ──────────                 ───────────              ─────────────────
    attn: (d, B*q_heads*S)     (B/KVDP, KVDP*q_heads, d, S)     (B, KVDP*q_heads, d, S)    (B, q_heads, d, S)       (d, B*q_heads*S) @ SBUF

**Example with KVDP=4, q_heads=1, B=8, S_tkg=1, d_head=64:**

    Input Collectives (B/KVDP=2, q_heads*KVDP=4):
        Q @ SBUF:  (64, 8)           - d_head=64, B*q_heads*S_tkg=8
        Q @ HBM:   (64, 8)           - no transpose needed (q_heads=1 optimization)
        Gathered:  (256, 8)          - KVDP*d_head=256, B*S_tkg=8
        Sliced:    (4, 64, 2, 1)     - KVDP*q_heads=4, d_head=64, B/KVDP=2, S_tkg=1
        Q @ SBUF:  (64, 8)           - d_head=64, B*q_heads*S_tkg=8

    Output Collectives:
        attn @ SBUF: (64, 8)         - d_head=64, B*q_heads*S_tkg=8
        attn @ HBM:  (2, 4, 64, 1)   - B/KVDP=2, KVDP*q_heads=4, d_head=64, S_tkg=1
        Gathered:    (8, 4, 64, 1)   - B=8, KVDP*q_heads=4, d_head=64, S_tkg=1
        Sliced:      (8, 1, 64, 1)   - B=8, q_heads=1, d_head=64, S_tkg=1
        attn @ SBUF: (64, 8)         - d_head=64, B*q_heads*S_tkg=8

**Example with KVDP=4, q_heads=2, B=8, S_tkg=1, d_head=64 (q_heads>1 path):**

    Input Collectives (B_attn=B/KVDP=2, q_heads_attn=KVDP*q_heads=8):
        Q @ SBUF:  (64, 16)          - (d=64, B*q*S=8*2*1=16)
        Rearrange: (64, 16)          - rearrange SBUF (d, B, q, S)=(64,8,2,1) → (d, q, B, S)=(64,2,8,1)
        Q @ HBM:   (2, 8, 1, 64)     - tiled transpose: (q=2, B=8, S=1, d=64)
        Gathered:  (8, 8, 1, 64)     - all_gather dim=0: (KVDP*q=8, B=8, S=1, d=64)
        Sliced:    (8, 2, 1, 64)     - KVDP_rank slice on batch: (KVDP*q=8, B_attn=2, S=1, d=64)
        Transpose: (64, 16)          - tiled transpose back: (d=64, q_attn*B_attn*S=8*2*1=16)
        Rearrange: (64, 16)          - rearrange SBUF (d, q_attn, B_attn, S)=(64,8,2,1) → (d, B_attn, q_attn, S)=(64,2,8,1)
        Q @ SBUF:  (64, 16)          - (d=64, B_attn*q_attn*S=16)

    Output Collectives:
        attn @ SBUF: (64, 16)        - (d=64, B_attn*q_attn*S=2*8*1=16)
        attn @ HBM:  (2, 8, 64, 1)   - tiled transpose: (B_attn=2, q_attn=8, d=64, S=1)
        Gathered:    (8, 8, 64, 1)   - all_gather dim=0: (B=8, q_attn=8, d=64, S=1)
        Sliced:      (8, 2, 64, 1)   - KVDP_rank slice on heads: (B=8, q=2, d=64, S=1)
        attn @ SBUF: (64, 16)        - tiled transpose back: (d=64, B*q*S=8*2*1=16)

See `_KVDP_attention_input_collectives` and `_KVDP_attention_output_collectives` docstrings for pseudocode.


## Future Optimizations

1. **KV projection for batch slice**: Currently each rank computes Q, K, V for all B batches (fused QKV kernel), then slices K/V to B/KVDP for cache update. The extra K/V compute is small relative to attention and avoids unfusing the QKV projection. 

2. **SB2SB collectives**: Both modes round-trip through HBM for collectives (SBUF → HBM → collective → HBM → SBUF). NKI collectives support SBUF tensors as src/dst, which would eliminate the HBM round-trips and reduce latency.

## Context Parallelism (Future)

Context parallelism shards KV cache along the sequence dimension instead of batch dimension.

    K/V cache shape: [n_layers, batch, kv_heads, s_prior/CP, d_head]
                                                 ^^^^^^^^^^^
                                                 context parallelism (CP)

Key differences from KV data parallelism:
- Each rank maintains full batches but only s_prior/CP sequence length
- For block KV: each rank loads only block_len/CP elements per block (element-level sharding)
- Requires distributed softmax correction across CP ranks
- Only rank 0 includes k_active/v_active to avoid counting active tokens CP times

Context parallelism is recommended when B < KVDP.
