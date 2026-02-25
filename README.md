# FP8 → BF16 Model Conversion for Apple Silicon

## What is Quantization?

Quantization is the process of reducing the numerical precision of a model's weights to save memory and speed up inference. A full-precision model stores each weight as a 32-bit float (`float32`). Quantizing it to 8-bit (`FP8`) cuts memory usage by ~75%, making large models practical to run on consumer hardware. The trade-off is a small loss in precision, which is usually negligible for image generation.

---

## The Problem: FP8 and Apple Silicon

FP8 quantized models come in several variants. The most common format used by Black Forest Labs (and others) is `float8_e4m3fn`. This format is **not natively supported by PyTorch's MPS backend** (the GPU backend for Apple Silicon). Running an FP8 model through ComfyUI on an M-series Mac will typically result in a `dtype` error, or if it runs at all, produce pure noise.

The naive fix — simply casting the weights to `float16` or `bfloat16` — does not work either, for two reasons:

1. **Scale factors are ignored.** Modern FP8 quantization does not just truncate values; it stores a separate `weight_scale` tensor for each layer. The FP8 values only have their correct magnitude when multiplied by this scale. A raw dtype cast throws the scales away entirely, leaving the weights at wildly wrong values — hence the noise.

2. **BF16, not FP16.** Most diffusion models (including all Flux variants) are trained in `bfloat16`. BF16 and FP16 have the same bit-width but different value ranges: FP16 has higher precision but a much narrower range, while BF16 matches the range of `float32`. Casting weights that were originally BF16 into FP16 can cause overflow and underflow, again producing noise.

---

## How the Conversion Script Works

The script performs a proper dequantization in three steps for each FP8 weight tensor:

1. **Find the scale.** For a weight stored as `some_layer.weight` (dtype `float8_e4m3fn`), the script looks for a corresponding `some_layer.weight_scale` tensor (dtype `float32`, scalar value).

2. **Dequantize via float32.** The FP8 tensor is first cast to `float32` as an intermediate step, then multiplied by its scale factor. This restores the original magnitude of the weights.

3. **Cast to BF16.** The correctly scaled `float32` tensor is then cast to `bfloat16`, which is the native precision these models expect.

All `weight_scale` and `input_scale` tensors are dropped from the output — they are only needed at inference time when running in FP8 mode, which the converted BF16 model no longer does.

---

## Troubleshooting: Inspecting a Model Before Converting

Before running the full conversion, it is worth inspecting the model to understand its structure. This is especially useful if the conversion produces warnings or unexpected results, or if you want to verify that a model uses per-layer scaling (which is required for a correct dequantization) rather than a simple FP8 cast.

The check_scales.py script is a read-only diagnostic tool — it does not modify or write any files. Run it on your original FP8 file to answer three questions:

1. **What dtypes are present?** If you see `float8_e4m3fn`, the model is FP8 and needs proper dequantization. If you see `uint8`, your PyTorch version doesn't recognize the FP8 dtype natively (see the Requirements section).
2. **Are scale tensors present?** If `Scale keys found` is 0, the model uses unscaled FP8 and a plain cast may suffice. If it's non-zero (as with all BFL Flux models), scales must be applied — skipping them is the most common cause of noise output.
3. **Do the scale key names match the pattern?** The conversion script expects scale keys to follow the naming convention `<weight_key>_scale` (e.g. `double_blocks.0.img_attn.proj.weight_scale`). If your model uses a different convention, the conversion script will fall through to plain casts and produce noise. The diagnostic output makes this immediately visible.

A healthy model ready for conversion will show output like this:

```
Total keys:        999
Scale keys found:  999 
```

---

## Usage

```bash
python convert_fp8_to_bf16.py <input_model.safetensors>
```

The script will produce a new file named `<input_model>-bf16.safetensors` in the same directory.

In ComfyUI, load the converted file using the **Load Diffusion Model** node and set the dtype to **default**.

---

## Compatibility

This script in theory works for any `safetensors` model that:

- Uses `float8_e4m3fn` quantization (the most common FP8 format)
- Stores per-layer scale factors as `<layer_name>.weight_scale` and `<layer_name>.input_scale` tensors alongside the weights

**Known compatible models:**
- FLUX.2 [klein] 4B FP8 variants - tested with [black-forest-labs/FLUX.2-klein-4b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8)
- FLUX.2 [klein] 9B FP8 - My experience is that the 9B model does not need the conversion
- Not tested, but in theory any Flux-family model quantized using the same BFL quantization pipeline

**Not compatible:**
- Models quantized with GGUF format (use `llama.cpp` tooling instead)
- Models using `float8_e5m2` (a different FP8 variant — the `view()` reinterpretation in the uint8 fallback branch would need updating)
- GPTQ, AWQ, or other integer quantization schemes

---

## Requirements

```
torch>=2.1.0
safetensors>=0.4.0
```

PyTorch 2.1+ is required for `torch.float8_e4m3fn` to be a recognized dtype. Older versions will load FP8 tensors as `uint8`, which the script handles via the fallback branch, but upgrading is recommended.
