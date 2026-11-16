# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unsplit BF16 wgrad: (K,C) output tiles, L=NZPQ reduction, one CTA per tile.

Each MN-major operand stage consists of two 64-channel by 64-pixel panels.
Within a panel the channel is contiguous (128 bytes, SW128); consecutive
pixels are 128 bytes apart. Thus the MMA descriptor's leading offset is
64*128=8192 bytes, its stride offset is 8*128=1024, and a K16 instruction
advances 16*128=2048 bytes. Both operands use this layout independently of
fprop's K-major layout. X uses im2col traversal; dY is a flat (L,K) tiled map.

Stage lifetimes: operand slots are released by tcgen05.commit after MMA;
TMEM slots by all 128 epilogue lanes, once per 32-column fragment, through
consume_then_arrive; store slots by bulk wait_group.read before reuse.
"""

from functools import lru_cache

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass.experimental import primitives as prims
from cutlass import utils
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

from cudnn.frost.device import device_context
from cudnn.frost.tile_dsl.consume import consume_then_arrive, release_zero
from .._cutedsl import build_target


@cute.kernel
def _kernel(
    params: utils.PersistentTileSchedulerParams,
    reduction_tiles: cutlass.Int32,
    zpq: tuple[cute.FastDivmodDivisorV2, cute.FastDivmodDivisorV2, cute.FastDivmodDivisorV2],
    rs: tuple[cute.FastDivmodDivisorV2, cute.FastDivmodDivisorV2],
    dy_map: cutlass.GridConstant[cuda.TensorMap],
    x_map: cutlass.GridConstant[cuda.TensorMap],
    dw_map: cutlass.GridConstant[cuda.TensorMap],
    release_arg: cutlass.Int32,
):
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tid, _, _ = cute.arch.thread_idx()
    rz = release_zero(release_arg)
    full = cutlass.Array(cutlass.Int64, 3, space=cutlass.AddressSpace.smem)
    empty = cutlass.Array(cutlass.Int64, 3, space=cutlass.AddressSpace.smem)
    acc_full = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem)
    acc_empty = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem)
    tmem_slot = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    a = cutlass.Array(cutlass.Int8, 3 * 16384, space=cutlass.AddressSpace.smem, alignment=1024)
    b = cutlass.Array(cutlass.Int8, 3 * 16384, space=cutlass.AddressSpace.smem, alignment=1024)
    out = cutlass.Array(cutlass.BFloat16, 2 * 128 * 32, space=cutlass.AddressSpace.smem, alignment=1024)
    if warp == 0:
        if prims.elect_sync():
            for i in cutlass.range_constexpr(3):
                prims.mbarrier_init(full.subview(i), 1)
                prims.mbarrier_init(empty.subview(i), 1)
            for i in cutlass.range_constexpr(2):
                prims.mbarrier_init(acc_full.subview(i), 1)
                prims.mbarrier_init(acc_empty.subview(i), 128 * 4)
    prims.fence_mbarrier_init()
    prims.barrier_cluster_arrive_relaxed()
    sched = utils.StaticPersistentTileScheduler.create(params, cute.arch.block_idx(), cute.arch.grid_dim())
    work = sched.initial_work_tile_info()
    prims.barrier_cluster_wait()

    if warp == 5:
        stage = 0
        phase = 1
        zdiv, pdiv, qdiv = zpq
        rdiv, sdiv = rs
        while work.is_valid_tile:
            kt, ct, f = work.tile_idx
            tr, s = divmod(f, sdiv)
            t, r = divmod(tr, rdiv)
            for step in cutlass.range(reduction_tiles, unroll=1):
                mb = full.subview(stage)
                while not prims.mbarrier_try_wait_parity(empty.subview(stage), phase, time_limit=10000000):
                    pass
                l = step * 64
                npz, q = divmod(l, qdiv)
                nz, p = divmod(npz, pdiv)
                n, z = divmod(nz, zdiv)
                if prims.elect_sync():
                    prims.mbarrier_arrive_expect_tx(mb, 32768)
                    for panel in cutlass.range_constexpr(2):
                        prims.cp_async_bulk_tensor_shared_cluster_global(
                            a.subview(stage * 16384 + panel * 8192),
                            dy_map.get_ptr(),
                            (kt * 128 + panel * 64, l),
                            mb,
                            [],
                            group=prims.CTAGroup.CTA_1,
                        )
                        prims.cp_async_bulk_tensor_shared_cluster_global(
                            b.subview(stage * 16384 + panel * 8192),
                            x_map.get_ptr(),
                            (ct * 128 + panel * 64, q, p, z, n),
                            mb,
                            [s, r, t],
                            mode=prims.TMALoadMode.IM2COL,
                            group=prims.CTAGroup.CTA_1,
                        )
                stage += 1
                if stage == 3:
                    stage = 0
                    phase ^= 1
            sched.advance_to_next_work()
            work = sched.get_current_work()
    elif warp == 4:
        prims.barrier_cta_sync(1, thread_count=160)
        tm = prims.make_tmem_ptr(tmem_slot.load(), cutlass.Float32)
        da = prims.Tcgen05SmemDesc.build(a, leading_byte_offset=8192, stride_byte_offset=1024, layout=2)
        db = prims.Tcgen05SmemDesc.build(b, leading_byte_offset=8192, stride_byte_offset=1024, layout=2)
        inst = prims.Tcgen05InstrDesc.build(
            a_dtype=cutlass.BFloat16,
            b_dtype=cutlass.BFloat16,
            c_dtype=cutlass.Float32,
            a_major=1,
            b_major=1,
            m_dim=128,
            n_dim=128,
        )
        stage, phase, acc, acc_phase = 0, 0, 0, 1
        while work.is_valid_tile:
            while not prims.mbarrier_try_wait_parity(acc_empty.subview(acc), acc_phase, time_limit=10000000):
                pass
            dst = cutlass.Array(tm.data_ptr() + acc * 128, dtype=cutlass.Int32, addrspace=6)
            for step in cutlass.range(reduction_tiles, unroll=1):
                while not prims.mbarrier_try_wait_parity(full.subview(stage), phase, time_limit=10000000):
                    pass
                for ki in cutlass.range_constexpr(4):
                    if prims.elect_sync():
                        prims.tcgen05_mma(
                            prims.Tcgen05MMAKind.F16,
                            prims.CTAGroup.CTA_1,
                            dst,
                            da + stage * 1024 + ki * 128,
                            db + stage * 1024 + ki * 128,
                            inst,
                            (step > 0) | (ki > 0),
                        )
                if prims.elect_sync():
                    prims.tcgen05_commit(empty.subview(stage), group=prims.CTAGroup.CTA_1)
                stage += 1
                if stage == 3:
                    stage = 0
                    phase ^= 1
            if prims.elect_sync():
                prims.tcgen05_commit(acc_full.subview(acc), group=prims.CTAGroup.CTA_1)
            acc += 1
            if acc == 2:
                acc = 0
                acc_phase ^= 1
            sched.advance_to_next_work()
            work = sched.get_current_work()
    elif warp < 4:
        if warp == 0:
            prims.tcgen05_alloc(tmem_slot, 256, group=prims.CTAGroup.CTA_1)
        prims.barrier_cta_sync(1, thread_count=160)
        raw = tmem_slot.load()
        acc, phase, store = 0, 0, 0
        rdiv, sdiv = rs
        while work.is_valid_tile:
            kt, ct, f = work.tile_idx
            tr, s = divmod(f, sdiv)
            t, r = divmod(tr, rdiv)
            while not prims.mbarrier_try_wait_parity(acc_full.subview(acc), phase, time_limit=10000000):
                pass
            for col in cutlass.range(4, unroll=1):
                tm = prims.make_tmem_ptr_from_warp_row_col(raw, warp, (raw & 0xFFFF) + acc * 128 + col * 32, cutlass.Float32)
                regs = prims.tcgen05_ld("32x32b", tm, num=32)
                sm = out.subview(store * 4096)
                for j in cutlass.range_constexpr(4):
                    sm.subview(tid * 32 + j * 8).data_ptr().store_swizzled(
                        regs[j * 8 : j * 8 + 8].to(cutlass.BFloat16),
                        alignment=64,
                        swizzle=cutlass.Swizzle(2, 4, 3),
                    )
                cute.arch.fence_view_async_shared()
                prims.barrier_cta_sync(2, thread_count=128)
                if warp == 0:
                    if prims.elect_sync():
                        prims.cp_async_bulk_tensor_global_shared_cta(dw_map.get_ptr(), sm, (ct * 128 + col * 32, s, r, t, kt * 128))
                    prims.cp_async_bulk_commit_group()
                    prims.cp_async_bulk_wait_group(1, read=True)
                prims.barrier_cta_sync(2, thread_count=128)
                consume_then_arrive(acc_empty.subview(acc), regs, rz)
                store ^= 1
            acc += 1
            if acc == 2:
                acc = 0
                phase ^= 1
            sched.advance_to_next_work()
            work = sched.get_current_work()
        if warp == 0:
            prims.cp_async_bulk_wait_group(0, read=True)
        prims.barrier_cta_sync(2, thread_count=128)
        if warp == 0:
            prims.tcgen05_relinquish_alloc_permit(group=prims.CTAGroup.CTA_1)
            prims.tcgen05_dealloc(prims.make_tmem_ptr(raw, cutlass.Float32), 256, group=prims.CTAGroup.CTA_1)


_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


class _WgradKernel:
    @cute.jit
    def __call__(self, x: cute.Tensor, dy: cute.Tensor, dw: cute.Tensor, max_ctas: cutlass.Constexpr, stream):
        n, z, p, q, k = dy.shape
        _, t, r, s, c = dw.shape
        length = n * z * p * q
        dy_flat = cute.make_tensor(dy.iterator, cute.make_layout((length, k), stride=(k, 1)))
        dy_map = cuda.create_tensor_map_tiled_from_view(dy_flat, box_dims=(64, 64), stride_order=(1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
        x_map = cuda.create_tensor_map_im2col_from_view(
            x,
            lower_corner=[0, 0, 0],
            upper_corner=[1 - s, 1 - r, 1 - t],
            channels_per_pixel=64,
            pixels_per_column=64,
            stride_order=(4, 3, 2, 1, 0),
            swizzle=cuda.TensorMapSwizzle.s128b,
        )
        dw_map = cuda.create_tensor_map_tiled_from_view(
            dw,
            box_dims=(128, 1, 1, 1, 32),
            stride_order=(4, 3, 2, 1, 0),
            swizzle=cuda.TensorMapSwizzle.s64b,
        )
        params = utils.PersistentTileSchedulerParams((k // 128, c // 128, t * r * s), (1, 1, 1))
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(params, max_ctas)
        _kernel(
            params,
            cutlass.Int32(length // 64),
            tuple(cute.fast_divmod_create_divisor_v2(cutlass.Int32(v)) for v in (z, p, q)),
            tuple(cute.fast_divmod_create_divisor_v2(cutlass.Int32(v)) for v in (r, s)),
            dy_map,
            x_map,
            dw_map,
            cutlass.Int32(0),
        ).launch(grid=grid, block=(192, 1, 1), cluster=(1, 1, 1), stream=stream, smem_merge_branch_allocs=True)


def compile():
    device, target = build_target()
    return _compile_cached(device, target, cutlass.BFloat16, (128, 128, 64, 1, 3, 2, 2), "identity")


@lru_cache(maxsize=None)
def _compile_cached(device, target, dtype, config, epilogue):
    # Shapes/geometry never enter this specialization key. All three descriptors
    # are rebuilt from symbolic metadata by the compiled host on the caller stream.
    with device_context(device):
        n, c, k = cute.sym_int(), cute.sym_int(divisibility=128), cute.sym_int(divisibility=128)
        shapes = [
            (n, cute.sym_int(), cute.sym_int(), cute.sym_int(), c),
            (n, cute.sym_int(), cute.sym_int(), cute.sym_int(), k),
            (k, cute.sym_int(), cute.sym_int(), cute.sym_int(), c),
        ]
        args = [make_fake_compact_tensor(dtype, shape, stride_order=(4, 3, 2, 1, 0), assumed_align=16) for shape in shapes]
        # Each CTA uses 256 TMEM columns and 115712 SMEM bytes including
        # alignment; two fit in the SM100-family SM. Keep enough persistent
        # CTAs to exercise both residents without specializing on work count.
        return cute.compile(
            _WgradKernel(), *args, 2 * utils.HardwareInfo().get_max_active_clusters(1), make_fake_stream(), options=f"--enable-tvm-ffi --gpu-arch {target}"
        )
