
# Convolutions

## Convolution Fprop

Convolution fprop computes:

$$ response = image * filter $$

### C++ API


```
std::shared_ptr<Tensor_attributes> conv_fprop(std::shared_ptr<Tensor_attributes> image,
                                                  std::shared_ptr<Tensor_attributes> filter,
                                                  Conv_fprop_attributes);
```

Conv_fprop_attributes is a lightweight structure with setters:  
```
Conv_fprop_attributes&
set_padding(std::vector<int64_t>)

Conv_fprop_attributes&
set_stride(std::vector<int64_t>)

Conv_fprop_attributes&
set_dilation(std::vector<int64_t>)

Conv_fprop_attributes&
set_name(std::string const&)

Conv_fprop_attributes&
set_compute_data_type(DataType_t value)

Conv_fprop_attributes&
set_convolution_mode(ConvolutionMode_t mode_)
```

### Python API

- conv_fprop
    - image
    - weight
    - pre_padding
    - post_padding
    - stride
    - dilation
    - convolution_mode
    - compute_data_type
    - name

The symmetric-padding overload accepts `padding` in place of `pre_padding` and `post_padding`.

## CuTe DSL Fused Conv3D + Post-Operations

The experimental direct Python APIs `Conv3dRmsNormSiluSm100` and
`Conv3dRmsNormSiluPadSm100` fuse Conv3D with bias, optional residual addition,
channel RMS normalization, affine scale, and SiLU. Related variants provide
raw Conv3D, bias/residual/spatial padding, and causal input packing plus Conv3D.
These APIs are not selected automatically by `cudnn.pygraph`.

### Current Limitations

- SM100 or SM103 GPUs, `nvidia-cutlass-dsl >= 4.9`, and BF16 inference only;
  no backward implementation.
- Fixed `3x3x3` convolution, unit stride and dilation, and one group. The
  valid-convolution APIs consume already-padded input when padding is needed.
- Selected channel pairs from the WAN VAE `base_dim=160` family (160/320/640),
  not arbitrary channels or the 128/256/512 family. Integrated normalization
  supports only `160->160`, `160->320`, and `320->320`; C640 output uses raw
  convolution followed by standalone normalization. The causal input variant
  currently supports only `12->160`. Unsupported configurations have no fallback.
- Contiguous NTHWC activations, except `CausalConv3dWithCacheSm100`, which accepts
  strided NCTHW input. Weights require variant-specific packing before execution.
- Normalization uses an L2-norm clamp followed by `sqrt(C)` scaling, with fixed
  BF16 rounding points; it is not arbitrary-epsilon RMSNorm.
- Padding/cache layouts are fixed for streaming convolution. The fused
  normalization-and-padding variant writes current frames and zeros but leaves
  existing-history interiors for the caller to fill.

See the [fused Conv3D + post-operations API reference](/operations/Convolutions/conv3d_postops)
for the complete per-variant channel table, tensor shapes, normalization math,
weight formats, and history/stream contracts.

## Convolution Dgrad

Convolution dgrad computes data gradient during backpropagation.

### C++ API

```
std::shared_ptr<Tensor_attributes> conv_dgrad(std::shared_ptr<Tensor_attributes> dy,
                                                  std::shared_ptr<Tensor_attributes> w,
                                                  Conv_dgrad_attributes);
```

Conv_dgrad_attributes is a lightweight structure with setters:  
```
Conv_dgrad_attributes&
set_padding(std::vector<int64_t>)

Conv_dgrad_attributes&
set_stride(std::vector<int64_t>)

Conv_dgrad_attributes&
set_dilation(std::vector<int64_t>)

Conv_dgrad_attributes&
set_name(std::string const&)

Conv_dgrad_attributes&
set_compute_data_type(DataType_t value)

Conv_dgrad_attributes&
set_convolution_mode(ConvolutionMode_t mode_)
```

### Python API

- conv_dgrad
    - loss
    - filter
    - pre_padding
    - post_padding
    - stride
    - dilation
    - convolution_mode
    - compute_data_type
    - name

`conv_dgrad` infers the dims of the returned dX from `loss`, `filter` and the padding/stride/dilation
attributes; per spatial axis the inferred extent is
`(S_loss - 1) * stride + (R - 1) * dilation + 1 - (pre_padding + post_padding)` (minus `2 * padding` for
the symmetric-padding overload). With `stride > 1` the forward extent division loses the remainder, so
the inferred dX can be smaller than the forward input even without padding: a 32x32 input with a 3x3
filter infers a 31x31 dX at stride 2 and padding 0, while stride 1 with padding 1 infers 32x32. Set the
dims and strides explicitly to get a dX shaped like the forward input:

```python
dX = graph.conv_dgrad(loss=dY, filter=W, padding=[1, 1], stride=[2, 2], dilation=[1, 1])
dX.set_output(True).set_dim(list(X.get_dim())).set_stride(list(X.get_stride()))
```

The symmetric-padding overload accepts `padding` in place of `pre_padding` and `post_padding`.

## Convolution Wgrad

Convolution wgrad computes weight gradient during backpropagation.

### C++ API

```
std::shared_ptr<Tensor_attributes> conv_wgrad(std::shared_ptr<Tensor_attributes> dy,
                                                  std::shared_ptr<Tensor_attributes> x,
                                                  Conv_wgrad_attributes);
```

Conv_wgrad_attributes is a lightweight structure with setters:  
```
Conv_wgrad_attributes&
set_padding(std::vector<int64_t>)

Conv_wgrad_attributes&
set_stride(std::vector<int64_t>)

Conv_wgrad_attributes&
set_dilation(std::vector<int64_t>)

Conv_wgrad_attributes&
set_name(std::string const&)

Conv_wgrad_attributes&
set_compute_data_type(DataType_t value)

Conv_wgrad_attributes&
set_convolution_mode(ConvolutionMode_t mode_)
```

### Python API

- conv_wgrad
    - image
    - loss
    - pre_padding
    - post_padding
    - stride
    - dilation
    - convolution_mode
    - compute_data_type
    - name

`conv_wgrad` cannot infer the dims of the returned dW from the graph inputs; set them explicitly
(the filter dims/strides) before marking it an output, otherwise graph construction fails with `dims not set`:

```python
dW = graph.conv_wgrad(image=X, loss=dY, padding=[1, 1], stride=[2, 2], dilation=[1, 1])
dW.set_output(True).set_dim(list(W.get_dim())).set_stride(list(W.get_stride()))
```

The symmetric-padding overload accepts `padding` in place of `pre_padding` and `post_padding`.

### Experimental FROST weight gradient

`frost_conv_wgrad` executes a single `graph.conv_wgrad(image=X, loss=dY, ...)`
through the existing graph planning and execution API. Enable experimental
engines with `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1`; this engine remains behind
backend plans in automatic ranking. Select its plan explicitly when qualifying
it. Forward convolution and weight gradients have separate engine families;
`frost_conv_wgrad` owns slot 0 of the weight-gradient family (engine ID 21200).
Public engine/knob records can be replayed with `create_execution_plan`.

The initial implementation accepts:

- SM100 or SM103 and CuTeDSL 4.9 or newer. SM100 has execution evidence;
  SM103 and other compiler versions/flavors remain unverified.
- BF16 X, dY and dW, with FP32 accumulation and identity epilogue. No fused
  pointwise nodes, grouped convolution, or convolution-mode filter reversal.
- Positive rank-5 compact channels-last tensors: logical NCDHW for X,
  NKZPQ for dY, and KCTRS for dW. Declare dW dimensions and strides explicitly.
- Zero padding and unit stride/dilation. Each filter extent is from 1 to 17,
  with positive output dimensions. The rank-5 im2col corners are
  `lower = -padding_before` and
  `upper = padding_after - (filter_extent - 1) * dilation`, both in `[-16, 15]`.
- C and K multiples of 128 and `L = N * Z * P * Q` divisible by 64. L is the
  reduction length, distinct from output-channel count K. Runtime addresses
  must be 16-byte aligned; descriptor and signed 32-bit scheduler bounds
  are checked before compilation.

The fixed public knobs are `TILE_M=128`, `TILE_N=128`, `TILEK=128` (bytes),
`CTA_GROUP=1`, and `STAGES=3`. Each CTA reduces all L elements for one
128-by-128 `(K, C)` tile and one filter position, with two accumulator stages
and two output-store stages. The persistent kernel uses tiled dY TMA loads,
im2col X TMA loads, MN-major MMA operands, and tiled dW TMA stores. Execution
launches one kernel on the caller's stream, needs zero workspace bytes, and
supports CUDA-graph capture.

Tensor dimensions and filter geometry are runtime values. Supported shapes
share one compiled artifact per device/target, dtype, pipeline configuration,
and epilogue. Unsupported configurations decline before JIT compilation.
