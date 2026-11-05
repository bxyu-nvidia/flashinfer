from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import *
from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import _pdl_wait_completed_before_tasks


# - This function describes part of one MLP layer: C = act(A * B + bias) where C[M, N] = A[M, K] × B[K, N]
# - This kernel function is called once for all tiles.
# - Usually the input tensor is [B, L, D] where B: Batch size, L: max sequence Length, D: hidden Dimension
#   but this is packed into [T, D] where T: Total tokens in this batch
# - Within one MLP layer, we call this function twice: once for fc1 and once for fc2, where e.g. fc1 = ReLU(W_1 * x)^2 and fc2 = W_2 fc1
#   - This translates to: fc1[D_upproj, T] = ReLU(W_1[D_upproj, D] * x[D, T])^2
# - For precisions lower than BF16, we have additional scale factor (sf) logic.
def _batched_gemm_kernel_bf16_body(
    # **Tensor memory accelerator (tma)** loads objects from global memory (global memory is shared across streaming multiprocessors or SMs)
    # into shared memory (memory shared by threads currently processed in an SM).
    # Specifically, it will load these objects tile by tile for the kernel to process.
    # `tma_a_desc` refers to the
    tma_a_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_b_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_c_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_sfa_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_sfb_desc: cutlass.GridConstant[cuda.TensorMap],
    c_tensor: cute.Tensor,
    sf_c_tensor: cute.Tensor,
    bias_tensor: cute.Tensor,
    scale_c_tensor: cute.Tensor,
    scale_gate_tensor: cute.Tensor,
    gemm1_alpha_tensor: cute.Tensor,
    gemm1_beta_tensor: cute.Tensor,
    gemm1_clamp_limit_tensor: cute.Tensor,
    per_token_sf_a_tensor: cute.Tensor,
    per_token_sf_b_tensor: cute.Tensor,
    tile_idx_tensor: cute.Tensor,
    route_map_tensor: cute.Tensor,
    mn_limit_tensor: cute.Tensor,
    num_non_exiting_ctas_tensor: cute.Tensor,
    total_num_padded_tokens_tensor: cute.Tensor,
    act_tensor: cute.Tensor,
    sfa_gmem_tensor: cute.Tensor,  # SF GMEM for LDGSTS SF (dummy when not LDGSTS)
    sfb_gmem_tensor: cute.Tensor,  # SF GMEM for LDGSTS SF (dummy when not LDGSTS)
    problem_m: cutlass.Int32,
    problem_n: cutlass.Int32,
    problem_k: cutlass.Int32,
    num_tokens: cutlass.Int32,
    num_experts: cutlass.Int32,
    tile_sched_params: object,
    cfg: cutlass.Constexpr[BatchedGemmConfig],
    early_exit_max_token_ctas: cutlass.Int32,
) -> None:
    """GPU kernel for BF16/FP4 BatchedGemm.

    tma_sfa_desc, tma_sfb_desc: SF TMA descriptors (FP4 only; dummy when BF16).
    route_map_tensor, mn_limit_tensor, act_tensor: gather only; dummy when off.
    """
    k_tile_cnt = (problem_k + Int32(cfg.tile_k - 1)) // Int32(cfg.tile_k)
    problem_m_tiles = (problem_m + Int32(cfg.tile_m - 1)) // Int32(cfg.tile_m)
    problem_n_tiles = (problem_n + Int32(cfg.tile_n - 1)) // Int32(cfg.tile_n)

    warp_idx = cute.arch.warp_idx()

    pcfgs = _make_pipeline_configs(cfg)

    num_non_exiting_ctas_value = None
    if cutlass.const_expr(
        cfg.is_persistent
        and cfg.use_early_exit
        and cfg.do_pdl_wait_for_num_non_exiting_ctas
        and not _pdl_wait_completed_before_tasks(cfg)
    ):
        num_non_exiting_ctas_view = cutlass.make_array_view(num_non_exiting_ctas_tensor)
        num_non_exiting_ctas_value = num_non_exiting_ctas_view.load(
            idx=Int32(0), vector_size=1
        )[0]

    tma_a_ptr = tma_a_desc.get_ptr()
    tma_b_ptr = tma_b_desc.get_ptr()
    tma_c_ptr = tma_c_desc.get_ptr()
    prims.prefetch_tensormap(tma_a_ptr)
    prims.prefetch_tensormap(tma_b_ptr)
    if cutlass.const_expr(cfg.has_scale_factors):
        prims.prefetch_tensormap(tma_sfa_desc.get_ptr())
        prims.prefetch_tensormap(tma_sfb_desc.get_ptr())
    prims.prefetch_tensormap(tma_c_ptr)

    # tile_idx_tensor: maps token tiles to expert indices.
    # mn_limit_tensor: maps token tiles to TRT-LLM Gen absolute end-row limits.
    tile_idx_view = cutlass.make_array_view(tile_idx_tensor)
    mn_limit_view = cutlass.make_array_view(mn_limit_tensor)
    rpe = (
        problem_m // num_experts
        if cutlass.const_expr(cfg.route_act == int(RouteImpl.TMA))
        else None
    )
    gmem_a = GmemAResource(
        cfg=cfg,
        tile_idx_view=tile_idx_view,
        mn_limit_view=mn_limit_view,
        rows_per_expert=rpe,
        name="GmemA",
    )
    gmem_b = GmemBResource(
        cfg=cfg,
        tile_idx_view=tile_idx_view,
        mn_limit_view=mn_limit_view,
        name="GmemB",
    )

    # Activation routing: LDGSTS gather or TMA gather4. Do not create a
    # default SMEM resource and then replace it inside a staged branch: PyIR
    # treats that as mutating a Python object from staged control flow.
    if cutlass.const_expr(cfg.has_gather and cfg.is_swap_ab):
        route_map_view = cutlass.make_array_view(route_map_tensor)
        mn_limit_view = cutlass.make_array_view(mn_limit_tensor)
        act_byte_ptr = cutlass.make_array_view(act_tensor).data_ptr()
        act_stride = problem_k * Int32(cfg.dtype_b_bits) // Int32(8)
        smem_a = SmemAResource(
            cfg=cfg,
            tma_a_desc=tma_a_ptr,
            pipeline_config=pcfgs["smem_a"],
            name="SmemA",
        )
        smem_b = SmemGatherResource(
            cfg=cfg,
            act_gmem_ptr=act_byte_ptr,
            act_stride_bytes=act_stride,
            route_map=route_map_view,
            mn_limit=mn_limit_view,
            pipeline_config=pcfgs["smem_gather"],
            _operand="b",
            name="SmemGather",
        )
    elif cutlass.const_expr(cfg.has_gather):
        route_map_view = cutlass.make_array_view(route_map_tensor)
        mn_limit_view = cutlass.make_array_view(mn_limit_tensor)
        act_byte_ptr = cutlass.make_array_view(act_tensor).data_ptr()
        act_stride = problem_k * Int32(cfg.dtype_a_bits) // Int32(8)
        smem_a = SmemGatherResource(
            cfg=cfg,
            act_gmem_ptr=act_byte_ptr,
            act_stride_bytes=act_stride,
            route_map=route_map_view,
            mn_limit=mn_limit_view,
            pipeline_config=pcfgs["smem_gather"],
            _operand="a",
            name="SmemGather",
        )
        smem_b = SmemBResource(
            cfg=cfg,
            tma_b_desc=tma_b_ptr,
            pipeline_config=pcfgs["smem_b"],
            name="SmemB",
        )
    elif cutlass.const_expr(cfg.has_tma_route and cfg.is_swap_ab):
        route_map_view = cutlass.make_array_view(route_map_tensor)
        mn_limit_view = cutlass.make_array_view(mn_limit_tensor)
        smem_a = SmemAResource(
            cfg=cfg,
            tma_a_desc=tma_a_ptr,
            pipeline_config=pcfgs["smem_a"],
            name="SmemA",
        )
        smem_b = SmemTmaGatherResource(
            cfg=cfg,
            tma_desc=tma_b_ptr,
            route_map=route_map_view,
            mn_limit=mn_limit_view,
            pipeline_config=pcfgs["smem_b"],
            _operand="b",
            name="SmemTmaGather",
        )
    elif cutlass.const_expr(cfg.has_tma_route):
        route_map_view = cutlass.make_array_view(route_map_tensor)
        mn_limit_view = cutlass.make_array_view(mn_limit_tensor)
        smem_a = SmemTmaGatherResource(
            cfg=cfg,
            tma_desc=tma_a_ptr,
            route_map=route_map_view,
            mn_limit=mn_limit_view,
            pipeline_config=pcfgs["smem_a"],
            _operand="a",
            name="SmemTmaGather",
        )
        smem_b = SmemBResource(
            cfg=cfg,
            tma_b_desc=tma_b_ptr,
            pipeline_config=pcfgs["smem_b"],
            name="SmemB",
        )
    else:
        smem_a = SmemAResource(
            cfg=cfg,
            tma_a_desc=tma_a_ptr,
            pipeline_config=pcfgs["smem_a"],
            name="SmemA",
        )
        smem_b = SmemBResource(
            cfg=cfg,
            tma_b_desc=tma_b_ptr,
            pipeline_config=pcfgs["smem_b"],
            name="SmemB",
        )

    tmem_c = TmemCResource(cfg=cfg, pipeline_config=pcfgs["tmem_c"], name="TmemC")
    tmem_sfa = None
    tmem_sfb = None
    tmem_sfab = None
    tmem_cast_a = None

    if cutlass.const_expr(cfg.has_cast_a):
        tma_sfa_ptr = tma_sfa_desc.get_ptr()
        gmem_sfa = GmemSfAResource(
            cfg=cfg,
            tile_idx_view=tile_idx_view,
            problem_m_tiles=problem_m_tiles,
            name="GmemSfA",
        )
        smem_sfa = SmemSfAResource(
            cfg=cfg,
            tma_sfa_desc=tma_sfa_ptr,
            pipeline_config=pcfgs["smem_sfa"],
            name="SmemSfA",
        )
        tmem_cast_a = TmemCastAResource(
            cfg=cfg,
            pipeline_config=pcfgs["tmem_cast_a"],
            name="TmemCastA",
        )

    # Scale factor resources (FP4/FP8 only)
    if cutlass.const_expr(cfg.has_scale_factors):
        tma_sfa_ptr = tma_sfa_desc.get_ptr()
        tma_sfb_ptr = tma_sfb_desc.get_ptr()
        gmem_sfa = GmemSfAResource(
            cfg=cfg,
            tile_idx_view=tile_idx_view,
            problem_m_tiles=problem_m_tiles,
            name="GmemSfA",
        )
        gmem_sfb = GmemSfBResource(
            cfg=cfg,
            tile_idx_view=tile_idx_view,
            problem_n_tiles=problem_n_tiles,
            name="GmemSfB",
        )

        # Routed operand's SF uses gather4 when route_sfs_act == TMA.
        # non-swapAB: A=activations (routed) → SFA gathered.
        # swapAB: B=activations (routed) → SFB gathered.
        if cutlass.const_expr(cfg.has_routed_sfs and cfg.uses_tma_routed_sfs):
            route_map_sf = cutlass.make_array_view(route_map_tensor)
            mn_limit_sf = cutlass.make_array_view(mn_limit_tensor)
            if cutlass.const_expr(cfg.is_swap_ab):
                smem_sfa = SmemSfAResource(
                    cfg=cfg,
                    tma_sfa_desc=tma_sfa_ptr,
                    pipeline_config=pcfgs["smem_sfa"],
                    name="SmemSfA",
                )
                smem_sfb = SmemSfGatherBResource(
                    cfg=cfg,
                    tma_sf_desc=tma_sfb_ptr,
                    route_map=route_map_sf,
                    mn_limit=mn_limit_sf,
                    pipeline_config=pcfgs["smem_sfb"],
                    name="SmemSfB",
                )
            else:
                smem_sfa = SmemSfGatherAResource(
                    cfg=cfg,
                    tma_sf_desc=tma_sfa_ptr,
                    route_map=route_map_sf,
                    mn_limit=mn_limit_sf,
                    pipeline_config=pcfgs["smem_sfa"],
                    name="SmemSfA",
                )
                smem_sfb = SmemSfBResource(
                    cfg=cfg,
                    tma_sfb_desc=tma_sfb_ptr,
                    pipeline_config=pcfgs["smem_sfb"],
                    name="SmemSfB",
                )
        elif cutlass.const_expr(cfg.has_routed_sfs and cfg.uses_ldgsts_routed_sfs):
            # LDGSTS/LDG+STS SF: per-thread loading with route map lookup.
            route_map_sf = cutlass.make_array_view(route_map_tensor)
            mn_limit_sf = cutlass.make_array_view(mn_limit_tensor)
            # SF GMEM stride is the number of scale-factor elements per row,
            # padded to 16-byte alignment. NVFP4 uses K/16; MX uses K/32.
            sf_stride_unpadded = problem_k // Int32(cfg.sf_vec_size)
            sf_stride = (sf_stride_unpadded + Int32(15)) // Int32(16) * Int32(16)
            if cutlass.const_expr(cfg.is_swap_ab):
                # SFB is routed (activations)
                sf_gmem_view = cutlass.make_array_view(sfb_gmem_tensor)
                smem_sfa = SmemSfAResource(
                    cfg=cfg,
                    tma_sfa_desc=tma_sfa_ptr,
                    pipeline_config=pcfgs["smem_sfa"],
                    name="SmemSfA",
                )
                smem_sfb = SmemSfLdgstsBResource(
                    cfg=cfg,
                    sf_gmem_ptr=sf_gmem_view,
                    sf_gmem_stride=sf_stride,
                    route_map=route_map_sf,
                    mn_limit=mn_limit_sf,
                    pipeline_config=pcfgs["smem_sfb"],
                    producer_commit_prefetch_depth=(
                        _ldgsts_sfb_producer_commit_prefetch_depth(cfg)
                    ),
                    name="SmemSfB",
                )
            else:
                # SFA is routed (activations)
                sf_gmem_view = cutlass.make_array_view(sfa_gmem_tensor)
                smem_sfa = SmemSfLdgstsAResource(
                    cfg=cfg,
                    sf_gmem_ptr=sf_gmem_view,
                    sf_gmem_stride=sf_stride,
                    route_map=route_map_sf,
                    mn_limit=mn_limit_sf,
                    pipeline_config=pcfgs["smem_sfa"],
                    name="SmemSfA",
                )
                smem_sfb = SmemSfBResource(
                    cfg=cfg,
                    tma_sfb_desc=tma_sfb_ptr,
                    pipeline_config=pcfgs["smem_sfb"],
                    name="SmemSfB",
                )
        else:
            smem_sfa = SmemSfAResource(
                cfg=cfg,
                tma_sfa_desc=tma_sfa_ptr,
                pipeline_config=pcfgs["smem_sfa"],
                name="SmemSfA",
            )
            smem_sfb = SmemSfBResource(
                cfg=cfg,
                tma_sfb_desc=tma_sfb_ptr,
                pipeline_config=pcfgs["smem_sfb"],
                name="SmemSfB",
            )
        # TmemSf resources: when SF is routed, separate CopySf tasks need pipelines.
        # When not routed, S2T is fused in MMA (pipeline_config=None).
        if cutlass.const_expr(cfg.use_combined_sfab_copy):
            tmem_sfab = TmemSfABResource(
                cfg=cfg,
                pipeline_config=pcfgs["tmem_sfab"],
                name="TmemSfAb",
            )
        elif cutlass.const_expr(cfg.uses_unfused_tmem_sf_copy):
            # Routed SF uses TmemSfRouteResource (LDS+STTM).
            # Non-routed SF uses standard TmemSfA/BResource (tcgen05_cp S2T).
            # Both get AsyncUmma pipeline for separate CopySf tasks.
            if cutlass.const_expr(cfg.is_swap_ab):
                # swapAB: SFB is routed (activations), SFA is not (weights)
                tmem_sfa = TmemSfAResource(
                    cfg=cfg, pipeline_config=pcfgs["tmem_sfa"], name="TmemSfA"
                )
                if cutlass.const_expr(
                    cfg.has_routed_sfs
                    or cfg.sfb_smem_to_tmem_copy == int(SfSmemToTmemCopy.LDS_STTM)
                ):
                    tmem_sfb = TmemSfRouteBResource(
                        cfg=cfg,
                        smem_sf_resource=smem_sfb,
                        pipeline_config=pcfgs["tmem_sfb"],
                        name="TmemSfB",
                    )
                else:
                    tmem_sfb = TmemSfBResource(
                        cfg=cfg, pipeline_config=pcfgs["tmem_sfb"], name="TmemSfB"
                    )
            else:
                # non-swapAB: SFA is routed (activations), SFB is not (weights)
                if cutlass.const_expr(cfg.has_routed_sfs):
                    tmem_sfa = TmemSfRouteAResource(
                        cfg=cfg,
                        smem_sf_resource=smem_sfa,
                        pipeline_config=pcfgs["tmem_sfa"],
                        name="TmemSfA",
                    )
                else:
                    tmem_sfa = TmemSfAResource(
                        cfg=cfg, pipeline_config=pcfgs["tmem_sfa"], name="TmemSfA"
                    )
                if cutlass.const_expr(
                    cfg.sfb_smem_to_tmem_copy == int(SfSmemToTmemCopy.LDS_STTM)
                ):
                    tmem_sfb = TmemSfRouteBResource(
                        cfg=cfg,
                        smem_sf_resource=smem_sfb,
                        pipeline_config=pcfgs["tmem_sfb"],
                        name="TmemSfB",
                    )
                else:
                    tmem_sfb = TmemSfBResource(
                        cfg=cfg, pipeline_config=pcfgs["tmem_sfb"], name="TmemSfB"
                    )
        else:
            # Not routed: fused S2T in MMA, no pipeline
            tmem_sfa = TmemSfAResource(cfg=cfg, pipeline_config=None, name="TmemSfA")
            tmem_sfb = TmemSfBResource(cfg=cfg, pipeline_config=None, name="TmemSfB")

    smem_dsfp8_sfab = None
    if cutlass.const_expr(cfg.has_deepseek_fp8):
        smem_dsfp8_sfab = SmemDeepSeekSfAbResource(
            cfg=cfg,
            sfa_gmem_base=cutlass.make_array_view(sfa_gmem_tensor),
            sfb_gmem_base=cutlass.make_array_view(sfb_gmem_tensor),
            tile_idx_view=tile_idx_view,
            mn_limit_view=mn_limit_view,
            route_map_view=cutlass.make_array_view(route_map_tensor),
            problem_m=problem_m,
            problem_n=problem_n,
            problem_k=problem_k,
            num_tokens=num_tokens,
            total_num_padded_tokens_tensor=total_num_padded_tokens_tensor,
            pipeline_config=pcfgs["smem_dsfp8_sfab"],
            name="SmemDeepSeekSfAb",
        )

    gmem_c = GmemCResource(
        cfg=cfg,
        c_tensor=c_tensor,
        sf_c_tensor=sf_c_tensor,
        tma_c_desc=tma_c_ptr,
        bias_tensor=bias_tensor,
        scale_c_tensor=scale_c_tensor,
        scale_gate_tensor=scale_gate_tensor,
        gemm1_alpha_tensor=gemm1_alpha_tensor,
        gemm1_beta_tensor=gemm1_beta_tensor,
        gemm1_clamp_limit_tensor=gemm1_clamp_limit_tensor,
        per_token_sf_a_tensor=per_token_sf_a_tensor,
        per_token_sf_b_tensor=per_token_sf_b_tensor,
        route_map_view=cutlass.make_array_view(route_map_tensor),
        tile_idx_view=tile_idx_view,
        mn_limit_view=mn_limit_view,
        total_num_padded_tokens_tensor=total_num_padded_tokens_tensor,
        problem_m=problem_m,
        problem_n=problem_n,
        name="GmemC",
    )

    # WorkQueue — CLC persistent or static (non-persistent)
    if cutlass.const_expr(cfg.is_persistent):
        if cutlass.const_expr(
            cfg.use_early_exit
            and (not cfg.use_pdl or _pdl_wait_completed_before_tasks(cfg))
        ):
            num_non_exiting_ctas_view = cutlass.make_array_view(
                num_non_exiting_ctas_tensor
            )
            num_non_exiting_ctas_value = num_non_exiting_ctas_view.load(
                idx=Int32(0), vector_size=1
            )[0]
        clc_response_ptr = cute.arch.alloc_smem(cutlass.Int128, cfg.num_stages_workid)
        tile_sched_cfg = (
            TileSchedulerConfig.create_clc_dynamic_persistent_tile_scheduler_params(
                tile_scheduler_params=tile_sched_params,
                response_ptr=clc_response_ptr,
            )
        )
        work_queue = BatchedGemmWorkQueue(
            tile_scheduler_config=tile_sched_cfg,
            cfg=cfg,
            num_non_exiting_ctas_tensor=num_non_exiting_ctas_tensor,
            num_non_exiting_ctas_value=num_non_exiting_ctas_value,
            pipeline_config=pcfgs["workid"],
            name="WorkQueue",
        )
    else:
        tile_sched_cfg = (
            TileSchedulerConfig.create_static_persistent_tile_scheduler_params(
                tile_scheduler_params=tile_sched_params,
            )
        )
        work_queue = WorkQueue(tile_scheduler_config=tile_sched_cfg, name="WorkQueue")

    work_throttle = None
    if cutlass.const_expr(cfg.use_work_throttle_barrier):
        work_throttle = WorkThrottleBarrierResource(
            pipeline_config=pcfgs["work_throttle"],
            name="WorkThrottle",
        )

    # SMEM allocator
    smem_allocator = SmemAllocator()
    smem_resources: tuple[Any, ...] = (smem_a, smem_b)
    if cutlass.const_expr(cfg.has_cast_a):
        smem_resources = smem_resources + (smem_sfa,)
    if cutlass.const_expr(cfg.has_scale_factors):
        smem_resources = smem_resources + (smem_sfa, smem_sfb)
    if cutlass.const_expr(cfg.has_deepseek_fp8):
        smem_resources = smem_resources + (smem_dsfp8_sfab,)
    if cutlass.const_expr(cfg.is_persistent):
        smem_resources = smem_resources + (work_queue,)
    smem_resources = smem_resources + (gmem_c,)
    for r in smem_resources:
        smem_allocator.add_resource(r)
    if cutlass.const_expr(cfg.aliases_c_scratch_with_ab):
        alloc_a = smem_a._alloc if hasattr(smem_a, "_alloc") else smem_a._alloc_a
        alloc_b = smem_b._alloc if hasattr(smem_b, "_alloc") else smem_b._alloc_b
        smem_allocator.add_alias_group(
            [
                [alloc_a, alloc_b],
                [gmem_c._alloc_sc],
            ]
        )
    tmem_ptr_alloc = smem_allocator.add_tmem_ptr(
        SmemAllocation("tmem_ptr_i32", dtype=cutlass.Int32, alignment=4)
    )
    tmem_dealloc_mbar_alloc = None
    if cutlass.const_expr(cfg.has_cluster):
        tmem_dealloc_mbar_alloc = smem_allocator.add(
            SmemAllocation("tmem_dealloc_mbar", dtype=cutlass.Int64, alignment=8)
        )
    smem_allocator.compute_layout()

    # TMEM allocator
    tmem_allocator = TmemAllocator()
    # C first (accumulator at offset 0, matching nvfp4_gemm reference)
    tmem_allocator.add_resource(tmem_c)
    if cutlass.const_expr(cfg.has_cast_a):
        tmem_allocator.add_resource(tmem_cast_a)
    if cutlass.const_expr(cfg.has_scale_factors and cfg.uses_unfused_tmem_sf_copy):
        if cutlass.const_expr(cfg.use_combined_sfab_copy):
            tmem_allocator.add_resource(tmem_sfab)
        else:
            tmem_allocator.add_resource(tmem_sfa)
            tmem_allocator.add_resource(tmem_sfb)
    tmem_allocator.compute_layout()

    pdl_wait_completed_before_tasks = cutlass.const_expr(
        _pdl_wait_completed_before_tasks(cfg)
    )
    pdl_wait_resource = (
        PdlWaitBarrier(name="PdlWait")
        if cutlass.const_expr(cfg.use_pdl and not pdl_wait_completed_before_tasks)
        else None
    )
    pdl_launch_resource = (
        PdlLaunchBarrier(name="PdlLaunch") if cutlass.const_expr(cfg.use_pdl) else None
    )

    # Tasks
    proxy_cluster = None
    if cutlass.const_expr(cfg.has_gather):
        if cutlass.const_expr(cfg.is_swap_ab):
            if cutlass.const_expr(cfg.fuse_operand_sf_loads):
                load_a = create_fused_load_a_sfa_task(
                    cfg,
                    gmem_a,
                    smem_a,
                    gmem_sfa,
                    smem_sfa,
                    work_queue,
                    k_tile_cnt,
                    work_throttle=work_throttle,
                )
                gather = create_fused_gather_sfb_task(
                    cfg,
                    gmem_b,
                    smem_b,
                    gmem_sfb,
                    smem_sfb,
                    work_queue,
                    k_tile_cnt,
                    pdl_wait_resource=pdl_wait_resource,
                    pdl_launch_resource=pdl_launch_resource,
                )
            else:
                load_a = create_load_a_task(
                    cfg,
                    gmem_a,
                    smem_a,
                    work_queue,
                    k_tile_cnt,
                    work_throttle=work_throttle,
                )
                gather = create_gather_task(
                    cfg,
                    gmem_b,
                    smem_b,
                    work_queue,
                    k_tile_cnt,
                    pdl_wait_resource=pdl_wait_resource,
                    pdl_launch_resource=pdl_launch_resource,
                )
            tma_smem = smem_a  # TMA resource for SyncTask
            gather_smem = smem_b  # Gather resource for SyncTask
            task_list = [load_a, gather]
        else:
            gather = create_gather_task(
                cfg,
                gmem_a,
                smem_a,
                work_queue,
                k_tile_cnt,
                pdl_wait_resource=pdl_wait_resource,
                pdl_launch_resource=pdl_launch_resource,
            )
            load_b = create_load_b_task(
                cfg,
                gmem_b,
                smem_b,
                work_queue,
                k_tile_cnt,
                work_throttle=work_throttle,
            )
            gather_smem = smem_a
            tma_smem = smem_b
            task_list = [gather, load_b]

        # 2-CTA + gather: add proxy barrier + sync task
        if cutlass.const_expr(cfg.has_cluster and cfg.num_sync_warps > 0):
            proxy_cluster = ProxyClusterBarrierResource(
                cfg=cfg,
                pipeline_config=pcfgs.get("proxy"),
                name="ProxyCluster",
            )
            sync = create_sync_task(
                cfg,
                proxy_cluster,
                gather_smem,
                tma_smem,
                work_queue,
                k_tile_cnt,
                sync_warp_idx=cfg.sync_warp_idx,
            )
            task_list.append(sync)
    else:
        load_a = create_load_a_task(
            cfg,
            gmem_a,
            smem_a,
            work_queue,
            k_tile_cnt,
            work_throttle=work_throttle,
            pdl_wait_resource=(
                None if cutlass.const_expr(cfg.is_swap_ab) else pdl_wait_resource
            ),
            pdl_launch_resource=(
                None if cutlass.const_expr(cfg.is_swap_ab) else pdl_launch_resource
            ),
        )
        load_b = create_load_b_task(
            cfg,
            gmem_b,
            smem_b,
            work_queue,
            k_tile_cnt,
            pdl_wait_resource=(
                pdl_wait_resource if cutlass.const_expr(cfg.is_swap_ab) else None
            ),
            pdl_launch_resource=(
                pdl_launch_resource if cutlass.const_expr(cfg.is_swap_ab) else None
            ),
        )
        task_list = [load_a, load_b]

    if cutlass.const_expr(cfg.has_cast_a):
        load_sfa = create_load_sfa_task(
            cfg,
            gmem_sfa,
            smem_sfa,
            work_queue,
            k_tile_cnt,
            pdl_wait_resource=(
                None if cutlass.const_expr(cfg.is_swap_ab) else pdl_wait_resource
            ),
            pdl_launch_resource=(
                None if cutlass.const_expr(cfg.is_swap_ab) else pdl_launch_resource
            ),
        )
        cast_a = create_cast_a_task(
            cfg,
            smem_a,
            smem_sfa,
            tmem_cast_a,
            work_queue,
            k_tile_cnt,
        )
        task_list += [load_sfa, cast_a]

    if cutlass.const_expr(cfg.has_scale_factors and not cfg.fuse_operand_sf_loads):
        load_sfa = create_load_sfa_task(
            cfg,
            gmem_sfa,
            smem_sfa,
            work_queue,
            k_tile_cnt,
            pdl_wait_resource=(
                None if cutlass.const_expr(cfg.is_swap_ab) else pdl_wait_resource
            ),
            pdl_launch_resource=(
                None if cutlass.const_expr(cfg.is_swap_ab) else pdl_launch_resource
            ),
        )
        load_sfb = create_load_sfb_task(
            cfg,
            gmem_sfb,
            smem_sfb,
            work_queue,
            k_tile_cnt,
            pdl_wait_resource=(
                pdl_wait_resource if cutlass.const_expr(cfg.is_swap_ab) else None
            ),
            pdl_launch_resource=(
                pdl_launch_resource if cutlass.const_expr(cfg.is_swap_ab) else None
            ),
        )
        task_list += [load_sfa, load_sfb]
        if cutlass.const_expr(cfg.uses_unfused_tmem_sf_copy):
            # Separate CopySf tasks for SMEM→TMEM when SF is routed.
            if cutlass.const_expr(cfg.use_combined_sfab_copy):
                copy_sfab = create_copy_sfab_task(
                    cfg, smem_sfa, smem_sfb, tmem_sfab, work_queue, k_tile_cnt
                )
                task_list.append(copy_sfab)
            else:
                copy_sfa = create_copy_sfa_task(
                    cfg, smem_sfa, tmem_sfa, work_queue, k_tile_cnt
                )
                copy_sfb = create_copy_sfb_task(
                    cfg, smem_sfb, tmem_sfb, work_queue, k_tile_cnt
                )
                task_list += [copy_sfa, copy_sfb]

    # With separate CopySf: MMA consumes TmemSf, not SmemSf.
    # Otherwise MMA consumes SmemSf and does fused S2T in producer_work.
    if cutlass.const_expr(cfg.has_scale_factors and cfg.uses_unfused_tmem_sf_copy):
        # MMA consumes TmemSf — SF already in TMEM from CopySf tasks
        smem_sfa_for_mma = None
        smem_sfb_for_mma = None
    elif cutlass.const_expr(cfg.has_scale_factors):
        # MMA consumes SmemSf — does fused S2T in producer_work
        smem_sfa_for_mma = smem_sfa
        smem_sfb_for_mma = smem_sfb
    else:
        smem_sfa_for_mma = None
        smem_sfb_for_mma = None
    # Pass TmemSf to MMA when separate CopySf tasks are enabled.
    tmem_sfa_for_mma = (
        tmem_sfa
        if cutlass.const_expr(
            cfg.has_scale_factors
            and cfg.uses_unfused_tmem_sf_copy
            and not cfg.use_combined_sfab_copy
        )
        else None
    )
    tmem_sfb_for_mma = (
        tmem_sfb
        if cutlass.const_expr(
            cfg.has_scale_factors
            and cfg.uses_unfused_tmem_sf_copy
            and not cfg.use_combined_sfab_copy
        )
        else None
    )
    tmem_sfab_for_mma = (
        tmem_sfab
        if cutlass.const_expr(cfg.has_scale_factors and cfg.use_combined_sfab_copy)
        else None
    )
    tmem_cast_a_for_mma = tmem_cast_a if cutlass.const_expr(cfg.has_cast_a) else None

    mma = create_mma_task(
        cfg,
        smem_a,
        smem_b,
        smem_sfa_for_mma,
        smem_sfb_for_mma,
        tmem_c,
        work_queue,
        k_tile_cnt,
        proxy_cluster=proxy_cluster,
        tmem_sfa=tmem_sfa_for_mma,
        tmem_sfb=tmem_sfb_for_mma,
        tmem_sfab=tmem_sfab_for_mma,
        tmem_cast_a=tmem_cast_a_for_mma,
    )
    if cutlass.const_expr(cfg.has_deepseek_fp8):
        load_sfab = create_load_sfab_task(
            cfg,
            smem_dsfp8_sfab,
            work_queue,
            k_tile_cnt,
            pdl_wait_resource=pdl_wait_resource,
            pdl_launch_resource=pdl_launch_resource,
        )
        epi = create_epilogue_task_dsfp8(
            cfg, tmem_c, smem_dsfp8_sfab, gmem_c, work_queue, k_tile_cnt
        )
        task_list += [mma, load_sfab, epi]
    else:
        epi = create_epilogue_task(cfg, tmem_c, gmem_c, work_queue, k_tile_cnt)
        task_list += [mma, epi]

    if cutlass.const_expr(cfg.num_padding_warps > 0):
        pad = create_padding_task(cfg, work_queue, k_tile_cnt)
        task_list.append(pad)
    # WorkScheduleTask (CLC persistent only)
    if cutlass.const_expr(cfg.is_persistent):
        workid = create_workid_task(
            cfg,
            work_queue,
            k_tile_cnt,
            work_throttle=work_throttle,
        )
        task_list.append(workid)

    resource_dependency_graph = {
        smem_a: [gmem_a],
        smem_b: [gmem_b],
        gmem_c: [tmem_c],
    }
    if cutlass.const_expr(cfg.has_scale_factors and cfg.uses_unfused_tmem_sf_copy):
        resource_dependency_graph[smem_sfa] = [gmem_sfa]
        resource_dependency_graph[smem_sfb] = [gmem_sfb]
        ab_dependencies = [smem_a, smem_b]
        if cutlass.const_expr(
            cfg.has_gather and cfg.has_cluster and cfg.num_sync_warps > 0
        ):
            resource_dependency_graph[proxy_cluster] = ab_dependencies
            ab_dependencies = [proxy_cluster]
        if cutlass.const_expr(cfg.use_combined_sfab_copy):
            resource_dependency_graph[tmem_sfab] = [smem_sfa, smem_sfb]
            resource_dependency_graph[tmem_c] = ab_dependencies + [tmem_sfab]
        else:
            resource_dependency_graph[tmem_sfa] = [smem_sfa]
            resource_dependency_graph[tmem_sfb] = [smem_sfb]
            # Separate CopySf: TmemC depends on the A/B readiness proxy (when
            # clustered gather is active) and both TMEM scale-factor rings.
            resource_dependency_graph[tmem_c] = ab_dependencies + [
                tmem_sfa,
                tmem_sfb,
            ]
    elif cutlass.const_expr(cfg.has_scale_factors):
        resource_dependency_graph[smem_sfa] = [gmem_sfa]
        resource_dependency_graph[smem_sfb] = [gmem_sfb]
        ab_dependencies = [smem_a, smem_b]
        if cutlass.const_expr(
            cfg.has_gather and cfg.has_cluster and cfg.num_sync_warps > 0
        ):
            resource_dependency_graph[proxy_cluster] = ab_dependencies
            ab_dependencies = [proxy_cluster]
        # Fused S2T+MMA waits for the clustered A/B readiness proxy (when
        # present) as well as both SMEM scale-factor rings.
        resource_dependency_graph[tmem_c] = ab_dependencies + [smem_sfa, smem_sfb]
    elif cutlass.const_expr(cfg.has_cast_a):
        resource_dependency_graph[smem_sfa] = [gmem_sfa]
        resource_dependency_graph[tmem_cast_a] = [smem_a, smem_sfa]
        if cutlass.const_expr(
            cfg.has_gather and cfg.has_cluster and cfg.num_sync_warps > 0
        ):
            resource_dependency_graph[proxy_cluster] = [smem_b]
            resource_dependency_graph[tmem_c] = [tmem_cast_a, proxy_cluster]
        else:
            resource_dependency_graph[tmem_c] = [tmem_cast_a, smem_b]
    elif cutlass.const_expr(
        cfg.has_gather and cfg.has_cluster and cfg.num_sync_warps > 0
    ):
        # 2-CTA+gather: MMA depends on proxy (which depends on gather+TMA)
        resource_dependency_graph[proxy_cluster] = [smem_a, smem_b]
        resource_dependency_graph[tmem_c] = [proxy_cluster]
    else:
        resource_dependency_graph[tmem_c] = [smem_a, smem_b]

    if cutlass.const_expr(cfg.has_deepseek_fp8):
        resource_dependency_graph[smem_dsfp8_sfab] = (
            [pdl_wait_resource]
            if cutlass.const_expr(pdl_wait_resource is not None)
            else []
        )
        resource_dependency_graph[gmem_c] = [tmem_c, smem_dsfp8_sfab]

    if cutlass.const_expr(cfg.is_persistent):
        for k in list(resource_dependency_graph.keys()):
            resource_dependency_graph[k].append(work_queue)
        if cutlass.const_expr(work_throttle is not None):
            throttle_producer = (
                smem_a
                if cutlass.const_expr(cfg.is_swap_ab or not cfg.has_gather)
                else smem_b
            )
            resource_dependency_graph[work_throttle] = [throttle_producer]
            resource_dependency_graph[work_queue] = [work_queue, work_throttle]
    if cutlass.const_expr(pdl_launch_resource is not None):
        resource_dependency_graph[pdl_launch_resource] = []
    if cutlass.const_expr(pdl_wait_resource is not None):
        resource_dependency_graph[pdl_wait_resource] = []
        if cutlass.const_expr(cfg.is_swap_ab):
            resource_dependency_graph[gmem_b] = [pdl_wait_resource]
            if cutlass.const_expr(cfg.has_scale_factors):
                resource_dependency_graph[gmem_sfb] = [pdl_wait_resource]
        else:
            resource_dependency_graph[gmem_a] = [pdl_wait_resource]
            if cutlass.const_expr(cfg.has_cast_a or cfg.has_scale_factors):
                resource_dependency_graph[gmem_sfa] = [pdl_wait_resource]
    task_manager_cls = (
        TaskManager if _task_manager_verify_enabled() else _ProductionTaskManager
    )
    task_manager = task_manager_cls(
        tasks=task_list,
        resource_dependency_graph=resource_dependency_graph,
        smem_allocator=smem_allocator,
        tmem_allocator=tmem_allocator,
        assume_pdl_wait_completed=pdl_wait_completed_before_tasks,
        exhaustive_deadlock_race_check=cutlass.const_expr(
            _exhaustive_deadlock_race_check_enabled()
        ),
    )

    # Early exit for dynamic-batch kernels: a max token-CTA grid is launched
    # for CUDA graph reuse and inactive CTAs skip execution.  The guard sits
    # here -- below resource/task construction -- so allocator registration
    # (trace-time layout bookkeeping) stays outside runtime control flow.
    if cutlass.const_expr(cfg.use_early_exit and not cfg.is_persistent):
        num_non_exiting_ctas_view = cutlass.make_array_view(num_non_exiting_ctas_tensor)
        num_non_exiting_ctas = num_non_exiting_ctas_view.load(
            idx=Int32(0), vector_size=1
        )[0]
        block_m, block_n, _ = cute.arch.block_idx()
        if cutlass.const_expr(cfg.is_swap_ab):
            token_cta_idx = block_n
        else:
            token_cta_idx = block_m
        is_active_cta = token_cta_idx < num_non_exiting_ctas
    else:
        is_active_cta = True

    if is_active_cta:
        # Setup
        task_manager.setup_resources_and_tasks()

        # Fence all mbarrier initializations before
        # any pipeline wait/arrive.  ``PipelineConfig`` creation uses
        # ``defer_sync=True``, so the common fence/sync has to happen here after
        # every resource has initialized its barriers.
        tmem_ptr_i32 = smem_allocator.get(tmem_ptr_alloc)
        tmem_dealloc_mbar = None
        tmem_dealloc_mbar_ptr = None
        if cutlass.const_expr(cfg.has_cluster):
            tmem_dealloc_mbar = smem_allocator.get(tmem_dealloc_mbar_alloc)
            tmem_dealloc_mbar_ptr = cute.make_ptr(
                cutlass.Int64,
                tmem_dealloc_mbar.data_ptr(),
                cutlass.AddressSpace.smem,
            )
            if warp_idx == 0:
                if prims.elect_sync():
                    cute.arch.mbarrier_init(tmem_dealloc_mbar_ptr, cute.arch.WARP_SIZE)
            cute.arch.mbarrier_init_fence()
        tmem_ptr_smem = cute.make_ptr(
            cutlass.Int32,
            tmem_ptr_i32.data_ptr(),
            cutlass.AddressSpace.smem,
        )
        cluster_vmnk = cute.make_layout((cfg.cluster_m, 1, 1, 1))
        pipeline.pipeline_init_arrive(cluster_vmnk, is_relaxed=True)
        pipeline.pipeline_init_wait(cluster_vmnk)

        # TMEM allocation
        # Allocate enough columns for both the explicit TS layout and generated
        # kernel fixed offsets. cfg.tmem_total_cols is derived from KernelTraits-style
        # formulas and includes the tile256 max-overlap lower bound.
        num_tmem_cols = _round_up_tmem_columns(
            max(tmem_allocator.total_tmem_columns, cfg.tmem_total_cols)
        )
        if warp_idx == 0:
            cute.arch.alloc_tmem(
                num_tmem_cols,
                tmem_ptr_smem,
                is_two_cta=cfg.has_cluster,
            )
            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=cfg.has_cluster)

        # Sync warps that need TMEM
        prims.barrier_cta_sync()

        # Run
        task_manager.run()

        # TMEM deallocation.  Only the epilogue warpgroup is synchronized
        # before deallocating from the first epilogue warp.  For CTA_2 dealloc, the
        # first epilogue warp from each peer CTA performs the 32-thread rendezvous
        # before issuing the collective dealloc.
        if cutlass.const_expr(cfg.has_cluster):
            epilogue_end = cfg.epilogue_warp_idx + cfg.num_epilogue_warps
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            if (warp_idx >= Int32(cfg.epilogue_warp_idx)) & (
                warp_idx < Int32(epilogue_end)
            ):
                prims.barrier_cta_sync(
                    barrier_id=10,
                    thread_count=cfg.num_epilogue_warps * 32,
                )
                if warp_idx == Int32(cfg.epilogue_warp_idx):
                    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
                    peer_cta_rank = cta_rank_in_cluster ^ 1
                    cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr, peer_cta_rank)
                    cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                    tmem_ptr_raw = prims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=Int32(tmem_ptr_i32.load()),
                        offset=0,
                        mask_and_clamp=0x1F,
                        kind=prims.Shfl.IDX,
                        return_value_and_is_valid=False,
                    )
                    tmem_ptr = prims.make_tmem_ptr(tmem_ptr_raw, cutlass.Float32)
                    prims.tcgen05_dealloc(tmem_ptr, num_tmem_cols, group=cfg.cta_group)
        else:
            epilogue_end = cfg.epilogue_warp_idx + cfg.num_epilogue_warps
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            if cutlass.const_expr(cfg.use_full_tmem_dealloc_barrier):
                prims.barrier_cta_sync()
            if (warp_idx >= Int32(cfg.epilogue_warp_idx)) & (
                warp_idx < Int32(epilogue_end)
            ):
                prims.barrier_cta_sync(
                    barrier_id=10,
                    thread_count=cfg.num_epilogue_warps * 32,
                )
                if warp_idx == Int32(cfg.epilogue_warp_idx):
                    tmem_ptr_raw = prims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=Int32(tmem_ptr_i32.load()),
                        offset=0,
                        mask_and_clamp=0x1F,
                        kind=prims.Shfl.IDX,
                        return_value_and_is_valid=False,
                    )
                    tmem_ptr = prims.make_tmem_ptr(tmem_ptr_raw, cutlass.Float32)
                    prims.tcgen05_dealloc(tmem_ptr, num_tmem_cols, group=cfg.cta_group)
