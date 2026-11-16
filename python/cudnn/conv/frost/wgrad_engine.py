# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed BF16, unsplit convolution weight gradient for SM100/SM103."""

from math import prod

import cudnn
from cuda.bindings import driver as cuda
from cudnn.engines.base import BaseEngine, CompiledPlan
from cudnn.frost.device import build_device, current_device
from ._cutedsl import DENSE_CUTEDSL_MIN_VERSION
from .checks import (
    _check_cutedsl,
    _check_sm100,
    _check_5d_inputs,
    _check_channels_last_3d_layout,
    _check_conv_mode,
    _check_3d_conv_geometry,
    _check_matching_channels,
    _check_fp32_compute_type,
    _check_output_shapes,
)
from .graph_analyzer import analyze_wgrad
from .utils import _tuple_param, _ZERO, _ONE, _channels_last_3d_stride, _reordering_name


def fixed_knobs():
    k = cudnn.knob_type
    return {k.TILE_M: 128, k.TILE_N: 128, k.TILEK: 128, k.CTA_GROUP: 1, k.STAGES: 3}


class _WgradPlan(CompiledPlan):
    takes_variant_pack = True

    def __init__(self, artifact, analysis, device):
        self._compiled = artifact
        self._tensors = (analysis.image, analysis.loss, analysis.weight)
        self._shapes = tuple(tuple(t.dim) for t in self._tensors)
        self._device = device
        self._indices = None

    def execute(self, graph, pack, ctx):
        if current_device() != self._device:
            raise ValueError("frost_conv_wgrad: execution device differs from plan device")
        if self._indices is None:
            self._indices = [pack.index_of(t.uid) for t in self._tensors]
        operands = pack.operands(self._indices)
        for role, operand, shape in zip(("X", "dY", "dW"), operands, self._shapes):
            if tuple(operand.shape) != shape or tuple(operand.stride()) != _channels_last_3d_stride(shape):
                raise ValueError(f"frost_conv_wgrad: {role} must match the declared compact channels-last shape and stride")
            if operand.dtype != "bfloat16" or operand.data_ptr() % 16:
                raise ValueError(f"frost_conv_wgrad: {role} requires BF16 storage and 16-byte alignment")
        self._compiled(*(v.permute(0, 2, 3, 4, 1) for v in operands), cuda.CUstream(ctx.stream or 0))


class FrostConvWgradEngine(BaseEngine):
    name = "frost_conv_wgrad"
    behavior_notes = (cudnn.behavior_note.RUNTIME_COMPILATION,)

    def check_support(self, graph):
        a = analyze_wgrad(graph)
        _check_cutedsl(self.name, DENSE_CUTEDSL_MIN_VERSION)
        _check_sm100()
        tensors = (("X", a.image), ("dY", a.loss), ("dW", a.weight))

        _check_5d_inputs(tensors)
        _check_channels_last_3d_layout(tensors)
        _check_matching_channels(a.image, a.weight)
        _check_conv_mode(a.conv_node)
        _check_fp32_compute_type(a.context)
        if a.conv_node.compute_data_type not in (None, cudnn.data_type.FLOAT):
            raise NotImplementedError("frost_conv_wgrad requires FP32 accumulation")
        if any(t.data_type != cudnn.data_type.BFLOAT16 for _, t in tensors):
            raise NotImplementedError("frost_conv_wgrad requires BF16 X, dY and dW")
        if any(_reordering_name(t) is not None for _, t in tensors):
            raise NotImplementedError("frost_conv_wgrad does not support reordered storage")
        if _tuple_param(a.conv_node, "padding", _ZERO) != _ZERO:
            raise NotImplementedError("frost_conv_wgrad requires zero padding")
        for name, default in (("pre_padding", _ZERO), ("post_padding", _ZERO), ("stride", _ONE), ("dilation", _ONE)):
            if _tuple_param(a.conv_node, name, default) != default:
                raise NotImplementedError("frost_conv_wgrad requires zero padding and unit stride/dilation")
        expected = _check_3d_conv_geometry(a.conv_node, a.image, a.weight)
        _check_output_shapes(expected, (("dY", a.loss),))
        if a.image.dim[1] % 128 or a.weight.dim[0] % 128:
            raise NotImplementedError("frost_conv_wgrad requires C and K multiples of 128")
        length = prod((expected[0], *expected[2:]))
        if length % 64:
            raise NotImplementedError("frost_conv_wgrad requires L=N*Z*P*Q divisible by 64")
        # Device coordinates and the scheduler use signed Int32. TMA dimensions
        # and byte strides also have finite descriptor fields; reject before JIT.
        if length >= 2**31 or prod(a.weight.dim[2:]) * (a.weight.dim[0] // 128) * (a.image.dim[1] // 128) >= 2**31:
            raise NotImplementedError("frost_conv_wgrad exceeds Int32 reduction/scheduler bounds")
        for _, tensor in tensors:
            if any(v >= 2**31 for v in tensor.dim) or any(v * 2 >= 2**40 for v in tensor.stride):
                raise NotImplementedError("frost_conv_wgrad exceeds tensor-map dimensions or strides")

    def knobs_to_public(self, knobs):
        if knobs is not None:
            self.knobs_from_public(knobs)
        return fixed_knobs()

    def knobs_from_public(self, public):
        supported = fixed_knobs()
        if any(key not in supported or value != supported[key] or type(value) is not int for key, value in public.items()):
            raise ValueError(f"frost_conv_wgrad supports only the fixed configuration {supported}")
        return supported

    def build_plan(self, graph, plan, ctx=None):
        self.knobs_to_public(plan.knobs)
        a = analyze_wgrad(graph)
        handle = ctx.handle if ctx is not None else None
        device = handle.device.ordinal if hasattr(handle, "device") else None
        with build_device(device):
            self.check_support(graph)
            # Load compiler primitives only after the DSL version/support gate.
            from .templates.sm100_conv_wgrad import compile

            return _WgradPlan(compile(), a, current_device())


def FrostConvWgradEngines(ids: dict[str, int]) -> list[BaseEngine]:
    """The weight-gradient engines the manifest asked for, with assigned ids."""
    out = []
    for cls in (FrostConvWgradEngine,):
        if cls.name in ids:
            engine = cls()
            engine.engine_id = ids[cls.name]
            out.append(engine)
    return out
