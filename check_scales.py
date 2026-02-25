from safetensors.torch import load_file

path = "your_model.safetensors"  # change this
state_dict = load_file(path)

scale_keys = [k for k in state_dict.keys() if 'scale' in k.lower()]
print(f"Total keys: {len(state_dict)}")
print(f"Scale keys found: {len(scale_keys)}")
for k in scale_keys[:20]:  # show first 20
    print(f"  {k}: {state_dict[k].dtype} {state_dict[k].shape}")