from safetensors.torch import load_file, save_file
import torch
import json
import sys

if len(sys.argv) < 2:
    print("Usage: python convert_fp8_to_bf16.py <input_safetensors_file>")
    sys.exit(1)

path = sys.argv[1]

def read_safetensors_metadata(path):
    with open(path, 'rb') as f:
        header_size = int.from_bytes(f.read(8), 'little')
        header_json = f.read(header_size).decode('utf-8')
        header = json.loads(header_json)
        return header.get('__metadata__', {})

metadata = read_safetensors_metadata(path)
print(json.dumps(metadata, indent=4))

state_dict = load_file(path)

scale_keys = {k for k in state_dict.keys() if k.endswith('.weight_scale') or k.endswith('.input_scale')}
weight_keys = [k for k in state_dict.keys() if k not in scale_keys]

print(f"Total keys: {len(state_dict)}")
print(f"Scale keys found: {len(scale_keys)}")
print(f"Weight keys to convert: {len(weight_keys)}")

sd_converted = {}

for key in weight_keys:
    tensor = state_dict[key]

    if tensor.dtype == torch.float8_e4m3fn:
        # Key is e.g. "double_blocks.0.img_attn.proj.weight"
        # Scale is stored as "double_blocks.0.img_attn.proj.weight_scale"
        # So we strip nothing — just append _scale suffix to the full key
        weight_scale_key = key + '_scale'  # → "...proj.weight_scale" ✓

        if weight_scale_key in state_dict:
            scale = state_dict[weight_scale_key]
            tensor = tensor.to(torch.float32) * scale
            print(f"  {key}: dequantized with scale {scale.item():.6f}")
        else:
            print(f"  WARNING: {key} is FP8 but has no scale at '{weight_scale_key}', doing plain cast")
            tensor = tensor.to(torch.float32)

        tensor = tensor.to(torch.bfloat16)

    elif tensor.dtype == torch.uint8:
        weight_scale_key = key + '_scale'
        tensor = tensor.view(torch.float8_e4m3fn).to(torch.float32)
        if weight_scale_key in state_dict:
            tensor = tensor * state_dict[weight_scale_key]
        tensor = tensor.to(torch.bfloat16)
        print(f"  WARNING: {key} was uint8, reinterpreted and dequantized")

    else:
        tensor = tensor.to(torch.bfloat16)

    print(f"  {key}: {state_dict[key].dtype} -> {tensor.dtype}")
    sd_converted[key] = tensor

dropped_input = len([k for k in scale_keys if k.endswith('.input_scale')])
dropped_weight = len([k for k in scale_keys if k.endswith('.weight_scale')])
print(f"\nDropped {dropped_input} input_scale keys")
print(f"Dropped {dropped_weight} weight_scale keys (applied to weights)")

output_filename = path.replace(".safetensors", "") + "-fp16.safetensors"
save_file(sd_converted, output_filename, metadata={"format": "pt"})
print(f"\nSaved to: {output_filename}")