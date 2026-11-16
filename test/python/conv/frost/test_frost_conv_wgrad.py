# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Milestone 1 public graph coverage; dgrad and higher derivatives are unsupported."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import cudnn
from cutlass import utils as cutlass_utils

from cudnn.conv.frost import wgrad_engine as engine
from cudnn.conv.frost.utils import _channels_last_3d_stride
from cudnn.conv.frost.wgrad_engine import FrostConvWgradEngine, fixed_knobs
from cudnn.engines.base import ExecutionContext
from cudnn.engines.engine_ids import FROST_CONV_WGRAD_ID_BASE
from cudnn.engines.heuristics import BACKEND
from cudnn.engines.manifest import engine_for_id, engines_for, family_for, resolve_analyzer, resolve_heuristics
from cudnn.frost import buffers

from utils import make_input, requires_dense_cutedsl
from warm_stress import StressCase, assert_release_dependence, assert_warm_stress

pytestmark = pytest.mark.L0
requires_wgrad_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)), reason="requires SM100 or SM103"
)


def declaration(xshape, wshape, *, dtype=cudnn.data_type.BFLOAT16, padding=(0, 0, 0), stride=(1, 1, 1), dilation=(1, 1, 1), handle=None):
    yshape = (xshape[0], wshape[0], *(xshape[i] - wshape[i] + 1 for i in range(2, 5)))
    g = cudnn.pygraph(handle=handle, io_data_type=dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    x = g.tensor(dim=xshape, stride=_channels_last_3d_stride(xshape), data_type=dtype)
    dy = g.tensor(dim=yshape, stride=_channels_last_3d_stride(yshape), data_type=dtype)
    dw = g.conv_wgrad(x, dy, padding=padding, stride=stride, dilation=dilation)
    dw.set_output(True).set_data_type(dtype).set_dim(wshape).set_stride(_channels_last_3d_stride(wshape))
    return g, (x, dy, dw)


def reference(x, dy, wshape):
    # Independent explicit filter-position contraction in FP64, no conv backend.
    n, k, z, p, q = dy.shape
    result = torch.empty(wshape, device=x.device, dtype=torch.float64)
    for t in range(wshape[2]):
        for r in range(wshape[3]):
            for s in range(wshape[4]):
                patch = x[:, :, t : t + z, r : r + p, s : s + q].double()
                result[:, :, t, r, s] = torch.einsum("nkzpq,nczpq->kc", dy.double(), patch)
    return result.to(torch.bfloat16)


def prepare(monkeypatch, xshape, wshape, *, handle=None, seed=3):
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    g, (gx, gy, gw) = declaration(xshape, wshape, handle=handle)
    x, dy = make_input(xshape, torch.bfloat16, seed), make_input(tuple(gy.dim), torch.bfloat16, seed + 1)
    dw = torch.empty(wshape, dtype=torch.bfloat16, device="cuda", memory_format=torch.channels_last_3d)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    index = next((i for i in range(len(g.plans)) if g.get_plan_name_at_index(i).split("[")[0] == "frost_conv_wgrad"), None)
    assert index is not None, "frost_conv_wgrad declined the supported graph"
    g.select_plan(index)
    g.check_support()
    g.build_plans()
    assert g.get_workspace_size() == 0
    workspace = torch.empty(0, dtype=torch.uint8, device="cuda")
    return g, {gx: x, gy: dy, gw: dw}, workspace, (x, dy, dw)


@requires_dense_cutedsl
@requires_wgrad_gpu
@pytest.mark.parametrize(
    "xshape,wshape",
    [
        ((1, 128, 1, 8, 8), (128, 128, 1, 1, 1)),
        ((2, 256, 3, 6, 10), (256, 256, 2, 3, 3)),
        ((1, 128, 17, 1, 64), (128, 128, 17, 1, 1)),
        ((1, 128, 1, 17, 64), (128, 128, 1, 17, 1)),
        ((1, 128, 1, 4, 32), (256, 128, 1, 1, 17)),
        ((2, 128, 4, 7, 10), (128, 128, 1, 4, 3)),
    ],
)
def test_frost_wgrad_numerics(monkeypatch, xshape, wshape):
    g, pack, ws, (x, dy, dw) = prepare(monkeypatch, xshape, wshape)
    g.execute(pack, ws)
    want = reference(x, dy, wshape)
    torch.testing.assert_close(dw, want, atol=0, rtol=0)
    # Quantized random values above give exact sums; retain a general rounding case.
    x.normal_(std=0.1)
    dy.normal_(std=0.1)
    g.execute(pack, ws)
    assert torch.isfinite(dw).all()
    torch.testing.assert_close(dw, reference(x, dy, wshape), atol=0.001, rtol=0.008)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"dtype": cudnn.data_type.HALF}, "BF16"),
        ({"padding": (1, 0, 0)}, "zero padding"),
        ({"stride": (1, 2, 1)}, "unit stride"),
        ({"dilation": (1, 1, 2)}, "unit stride"),
        ({"xshape": (1, 64, 1, 8, 8), "wshape": (128, 64, 1, 1, 1)}, "multiples"),
        ({"wshape": (64, 128, 1, 1, 1)}, "multiples"),
        ({"xshape": (1, 128, 1, 8, 9)}, "divisible"),
        ({"xshape": (1, 128, 18, 1, 64), "wshape": (128, 128, 18, 1, 1)}, "corners"),
    ],
)
def test_frost_wgrad_rejections(monkeypatch, change, match):
    monkeypatch.setattr(engine, "_check_sm100", lambda: None)
    monkeypatch.setattr(engine, "_check_cutedsl", lambda *a: None)
    args = dict(xshape=(1, 128, 1, 8, 8), wshape=(128, 128, 1, 1, 1)) | change
    graph, _ = declaration(**args)
    with pytest.raises(NotImplementedError, match=match):
        engine.FrostConvWgradEngine().check_support(graph)


@requires_dense_cutedsl
@requires_wgrad_gpu
def test_frost_wgrad_dynamic_replay(monkeypatch):
    # Keep newer compiler primitives behind requires_dense_cutedsl during collection.
    from cudnn.conv.frost.templates import sm100_conv_wgrad as module

    module._compile_cached.cache_clear()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    handle = cudnn.create_handle()
    cudnn.set_stream(handle, stream.cuda_stream)
    retained = []
    try:
        with torch.cuda.stream(stream):
            for i, (xs, ws) in enumerate(
                [
                    ((1, 128, 1, 8, 8), (128, 128, 1, 1, 1)),
                    ((2, 256, 3, 6, 10), (256, 256, 2, 3, 3)),
                    ((1, 128, 17, 1, 64), (128, 128, 17, 1, 1)),
                ]
            ):
                g, pack, scratch, (x, dy, dw) = prepare(monkeypatch, xs, ws, handle=handle)
                index = next(j for j in range(len(g.plans)) if g.get_plan_name_at_index(j).split("[")[0] == "frost_conv_wgrad")
                eid, knobs = g.get_engine_and_knobs_at_index(index)
                assert eid == FROST_CONV_WGRAD_ID_BASE
                assert knobs == fixed_knobs()
                g.create_execution_plan(eid, knobs)
                g.select_plan(len(g.plans) - 1)
                g.build_plans()
                if i == 0:
                    artifact = module.compile()
                    monkeypatch.setattr(module.cute, "compile", lambda *a, **k: pytest.fail("shape change compiled another artifact"))
                assert module.compile() is artifact
                g.execute(pack, scratch)
                capture = torch.cuda.CUDAGraph()
                with torch.cuda.graph(capture, stream=stream):
                    g.execute(pack, scratch)
                x.mul_(0.5)
                capture.replay()
                torch.testing.assert_close(dw, reference(x, dy, ws), atol=0, rtol=0)
                retained.append((g, pack, scratch, capture))
            # Older captured host descriptors must remain independent of later shapes.
            for g, pack, scratch, capture in retained:
                x, dy, dw = pack.values()
                capture.replay()
                torch.testing.assert_close(dw, reference(x, dy, dw.shape), atol=0, rtol=0)
        stream.synchronize()
    finally:
        torch.cuda.current_stream().wait_stream(stream)
        cudnn.destroy_handle(handle)


@requires_dense_cutedsl
@requires_wgrad_gpu
def test_frost_wgrad_warm_refill(monkeypatch):
    g, pack, ws, (x, dy, dw) = prepare(monkeypatch, (2, 384, 5, 10, 10), (384, 384, 4, 3, 3))
    # 324 work items, L=256: operand ring wraps within each persistent item.
    out = dw.permute(0, 2, 3, 4, 1)
    want = reference(x, dy, dw.shape).permute(0, 2, 3, 4, 1).contiguous()
    assert_warm_stress(
        StressCase(
            "frost_conv_wgrad_bf16",
            lambda: g.execute(pack, ws),
            {"dw": out},
            min(324, 2 * cutlass_utils.HardwareInfo().get_max_active_clusters(1)),
            1,
            {"dw": want},
        )
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("layout", "channels-last"),
        ("rank", "rank-5"),
        ("channels", "channels differ"),
        ("loss", "dY shape"),
        ("compute", "FP32"),
        ("mode", "convolution mode"),
        ("epilogue", "identity epilogue"),
        ("extent", "positive"),
    ],
)
def test_frost_wgrad_graph_rejections(monkeypatch, mutation, match):
    monkeypatch.setattr(engine, "_check_sm100", lambda: None)
    monkeypatch.setattr(engine, "_check_cutedsl", lambda *a: None)
    graph, (x, dy, dw) = declaration((1, 128, 1, 8, 8), (128, 128, 1, 1, 1))
    if mutation == "layout":
        x.set_stride((8192, 64, 64, 8, 1))
    elif mutation == "rank":
        x.set_dim((1, 128, 8, 8))
    elif mutation == "channels":
        x.set_dim((1, 256, 1, 8, 8)).set_stride((16384, 1, 16384, 2048, 256))
    elif mutation == "loss":
        dy.set_dim((1, 128, 1, 4, 16)).set_stride((8192, 1, 8192, 2048, 128))
    elif mutation == "compute":
        graph.nodes[0].compute_data_type = cudnn.data_type.HALF
    elif mutation == "mode":
        graph.nodes[0].params["convolution_mode"] = cudnn._pybind_module.convolution_mode.CONVOLUTION
    elif mutation == "epilogue":
        graph.relu(dw).set_output(True)
    else:
        x.set_dim((0, 128, 1, 8, 8))
    with pytest.raises(NotImplementedError, match=match):
        engine.FrostConvWgradEngine().check_support(graph)


@pytest.mark.parametrize("version", ["4.8.0", "4.8.2"])
def test_frost_wgrad_dependency_gate(monkeypatch, version):
    graph, _ = declaration((1, 128, 1, 8, 8), (128, 128, 1, 1, 1))
    monkeypatch.setattr(buffers, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", version)))
    with pytest.raises(NotImplementedError, match=r">= 4.9"):
        engine.FrostConvWgradEngine().check_support(graph)


def test_frost_wgrad_knob_rejection():
    engine = FrostConvWgradEngine()
    assert engine.knobs_to_public(engine.knobs_from_public(fixed_knobs())) == fixed_knobs()
    for key in fixed_knobs():
        with pytest.raises(ValueError, match="fixed configuration"):
            engine.knobs_from_public(fixed_knobs() | {key: 99})
    with pytest.raises(ValueError):
        engine.knobs_from_public({cudnn.knob_type.SPLIT_K_SLC: 2})
    for value in (True, 1.0):
        with pytest.raises(ValueError):
            engine.knobs_from_public({cudnn.knob_type.CTA_GROUP: value})


@pytest.mark.parametrize("role", [0, 1, 2])
@pytest.mark.parametrize("field", ["shape", "stride", "dtype", "alignment"])
def test_frost_wgrad_binding_rejection(monkeypatch, role, field):
    graph, tensors = declaration((1, 128, 1, 8, 8), (128, 128, 1, 1, 1))
    analysis = engine.analyze_wgrad(graph)
    calls = []
    monkeypatch.setattr(engine, "current_device", lambda: 0)
    plan = engine._WgradPlan(lambda *a: calls.append(a), analysis, 0)
    operands = []
    for i, tensor in enumerate(tensors):
        stride = tuple(tensor.stride)
        v = SimpleNamespace(shape=tuple(tensor.dim), stride=lambda stride=stride: stride, dtype="bfloat16", data_ptr=lambda: 16)
        if i == role:
            if field == "shape":
                v.shape = (1,)
            elif field == "stride":
                v.stride = lambda: (1,) * 5
            elif field == "dtype":
                v.dtype = "float16"
            else:
                v.data_ptr = lambda: 17
        operands.append(v)
    indices = {t.uid: i for i, t in enumerate(tensors)}
    pack = SimpleNamespace(index_of=lambda uid: indices[uid], operands=lambda i: operands)
    with pytest.raises(ValueError):
        plan.execute(graph, pack, ExecutionContext(stream=0))
    assert not calls


@requires_dense_cutedsl
@requires_wgrad_gpu
def test_frost_wgrad_release_rendering(tmp_path):
    # Compile in a fresh process so the dump options precede the DSL import.
    source = "import cudnn\n" f"assert cudnn.__file__ == {cudnn.__file__!r}\n" "from cudnn.conv.frost.templates.sm100_conv_wgrad import compile\n" "compile()\n"
    env = dict(os.environ, CUTE_DSL_KEEP="ptx,cubin", CUTE_DSL_DUMP_DIR=str(tmp_path), CUTE_DSL_CACHE_DIR=str(tmp_path / "cache"))
    result = subprocess.run([sys.executable, "-c", source], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    renders = list(Path(tmp_path).glob("*.ptx"))
    assert len(renders) == 1
    assert_release_dependence(renders[0].read_text(), sites=1, words=32)


def test_frost_wgrad_slot_and_ranking(monkeypatch):
    graph, _ = declaration((1, 128, 1, 8, 8), (128, 128, 1, 1, 1))
    family = family_for(graph)
    assert family.name == "frost_conv_wgrad"
    assert family.engine_id == FROST_CONV_WGRAD_ID_BASE
    assert set(family.slots) == {"frost_conv_wgrad"}
    assert family.slots["frost_conv_wgrad"].slot == 0
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    assert engines_for(graph) == []
    assert engine_for_id(FROST_CONV_WGRAD_ID_BASE) is None
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    engines = engines_for(graph)
    assert [(e.name, e.engine_id) for e in engines] == [("frost_conv_wgrad", FROST_CONV_WGRAD_ID_BASE)]
    assert engine_for_id(FROST_CONV_WGRAD_ID_BASE) is engines[0]
    facts = resolve_analyzer(family)(graph)
    candidates = resolve_heuristics(family)("A", facts, family.offered_ids())
    assert candidates[0] is BACKEND
    assert candidates[1].engine_id == FROST_CONV_WGRAD_ID_BASE
