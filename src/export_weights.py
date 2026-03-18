import h5py
import numpy as np
import json
import os

h5_path = 'export_6classes/attack_multiclass.h5'
output_path = 'model_weights.h'

def format_array(name, data, is_1d=False):
    data = np.array(data)
    if is_1d:
        values = ", ".join(f"{v:.6f}f" for v in data)
        return f"const float {name}[{data.shape[0]}] = {{ {values} }};\n"
    else:
        # 2D array: C++ expects row-major. Keras weights for Dense are (input_dim, output_dim)
        # We will keep them as (input_dim, output_dim) and handle it in C++ MatMul
        rows, cols = data.shape
        out = f"const float {name}[{rows}][{cols}] = {{\n"
        for r in range(rows):
            row_vals = ", ".join(f"{v:.6f}f" for v in data[r])
            out += f"    {{ {row_vals} }}"
            if r < rows - 1:
                out += ",\n"
            else:
                out += "\n"
        out += "};\n"
        return out

print(f"Reading weights from {h5_path}...")
try:
    with h5py.File(h5_path, 'r') as f:
        # --- Normalization ---
        # Note: Keras normalization layer usually has mean, variance, count.
        norm_mean = f['model_weights']['normalization_4']['mean:0'][:]
        norm_var = f['model_weights']['normalization_4']['variance:0'][:]
        
        # --- Dense 1 (64 units) ---
        d1_w = f['model_weights']['dense_8']['dense_8']['kernel:0'][:]
        d1_b = f['model_weights']['dense_8']['dense_8']['bias:0'][:]
        
        # --- Dense 2 (32 units) ---
        d2_w = f['model_weights']['dense_9']['dense_9']['kernel:0'][:]
        d2_b = f['model_weights']['dense_9']['dense_9']['bias:0'][:]
        
        # --- Output (10 units) ---
        d3_w = f['model_weights']['class_probs']['class_probs']['kernel:0'][:]
        d3_b = f['model_weights']['class_probs']['class_probs']['bias:0'][:]

    print("Writing to model_weights.h...")
    with open(output_path, 'w') as out_f:
        out_f.write("#ifndef MODEL_WEIGHTS_H\n")
        out_f.write("#define MODEL_WEIGHTS_H\n\n")
        
        out_f.write("// Auto-generated from attack_multiclass.h5\n\n")
        
        out_f.write(format_array("norm_mean", norm_mean, is_1d=True))
        out_f.write(format_array("norm_var", norm_var, is_1d=True))
        out_f.write("\n")
        
        out_f.write(format_array("W1_base", d1_w, is_1d=False))
        out_f.write(format_array("b1_base", d1_b, is_1d=True))
        out_f.write("\n")
        
        out_f.write(format_array("W2_base", d2_w, is_1d=False))
        out_f.write(format_array("b2_base", d2_b, is_1d=True))
        out_f.write("\n")
        
        out_f.write(format_array("W3_base", d3_w, is_1d=False))
        out_f.write(format_array("b3_base", d3_b, is_1d=True))
        out_f.write("\n")
        
        out_f.write("#endif // MODEL_WEIGHTS_H\n")
        
    print(f"Successfully generated {output_path}!")

except Exception as e:
    print(f"Error: {e}")
