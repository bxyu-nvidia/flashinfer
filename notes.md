
## Popular kernel optimization techniques
   Technique                            Category
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Loop unrolling                       Loop/compiler optimization
  ───────────────────────────────────  ─────────────────────────────────────────
   Multiple independent accumulators    Instruction-level parallelism
  ───────────────────────────────────  ─────────────────────────────────────────
   float4 loads/stores                  Memory vectorization
  ───────────────────────────────────  ─────────────────────────────────────────
   Coalesced memory access              Memory-access optimization
  ───────────────────────────────────  ─────────────────────────────────────────
   Shared-memory tiling                 Data-reuse optimization
  ───────────────────────────────────  ─────────────────────────────────────────
   Warp shuffles for reduction          Inter-thread communication optimization
  ───────────────────────────────────  ─────────────────────────────────────────
   Kernel fusion                        Reducing launches and memory traffic
  ───────────────────────────────────  ─────────────────────────────────────────
   CUDA Graphs                          Launch-overhead optimization
