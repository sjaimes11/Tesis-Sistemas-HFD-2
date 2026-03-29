"""
=============================================================================
 export_to_c.py — Exporta modelos sklearn a código C puro para ESP32
=============================================================================
 Usa m2cgen (model-to-code-generator) para convertir cualquier modelo
 de scikit-learn a funciones C puras, sin dependencias externas.
 
 También genera el scaler inline para que el ESP32 normalice las features.
 
 Uso:
   python export_to_c.py decision_tree_best.pkl scaler_dt.pkl
   python export_to_c.py logistic_regression_best.pkl scaler_lr.pkl
   python export_to_c.py random_forest_best.pkl         (sin scaler)
=============================================================================
"""

import sys
import os
import joblib
import numpy as np
import m2cgen as m2c

FEATURE_NAMES = [
    "num_pkts", "mean_iat", "std_iat", "min_iat", "max_iat",
    "mean_pkt_len", "num_bytes", "num_psh_flags", "num_rst_flags",
    "num_urg_flags", "std_pkt_len", "min_pkt_len", "max_pkt_len"
]

CLASS_NAMES = ["normal", "mqtt_bruteforce", "scan_A"]


def generate_scaler_code(scaler) -> str:
    """Genera arrays C con mean y scale del StandardScaler."""
    means = scaler.mean_
    scales = scaler.scale_
    lines = [
        f"#define NUM_FEATURES {len(means)}",
        "",
        "static const float scaler_mean[NUM_FEATURES] = {",
        "    " + ", ".join(f"{v:.8f}f" for v in means),
        "};",
        "",
        "static const float scaler_scale[NUM_FEATURES] = {",
        "    " + ", ".join(f"{v:.8f}f" for v in scales),
        "};",
        "",
        "void apply_scaler(const float raw[NUM_FEATURES], float scaled[NUM_FEATURES]) {",
        "    for (int i = 0; i < NUM_FEATURES; i++) {",
        "        scaled[i] = (raw[i] - scaler_mean[i]) / scaler_scale[i];",
        "    }",
        "}",
    ]
    return "\n".join(lines)


def generate_predict_wrapper(num_classes: int) -> str:
    """Genera wrapper que devuelve la clase con mayor score."""
    return f"""
#define NUM_CLASSES {num_classes}

int predict_class(const float features[NUM_FEATURES], float* confidence) {{
    double scores[NUM_CLASSES];
    score(features, scores);
    
    int best = 0;
    double best_score = scores[0];
    double sum_exp = 0.0;
    
    for (int i = 1; i < NUM_CLASSES; i++) {{
        if (scores[i] > best_score) {{
            best_score = scores[i];
            best = i;
        }}
    }}
    
    // Softmax para confidence
    for (int i = 0; i < NUM_CLASSES; i++) {{
        sum_exp += exp(scores[i] - best_score);
    }}
    *confidence = (float)(1.0 / sum_exp);
    
    return best;
}}

// Wrapper completo: raw features -> clase predicha
int classify(const float raw_features[NUM_FEATURES], float* confidence) {{
    float scaled[NUM_FEATURES];
    apply_scaler(raw_features, scaled);
    return predict_class(scaled, confidence);
}}
"""


def export_model(model_path: str, scaler_path: str = None):
    print(f"Cargando modelo: {model_path}")
    model = joblib.load(model_path)
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    print(f"Convirtiendo a código C con m2cgen...")
    c_code = m2c.export_to_c(model)

    output_h = f"{model_name}_esp32.h"

    header_guard = model_name.upper() + "_ESP32_H"
    lines = [
        f"#ifndef {header_guard}",
        f"#define {header_guard}",
        "",
        "/*",
        f" * Auto-generated from {os.path.basename(model_path)}",
        f" * Model: {type(model).__name__}",
        f" * Classes: {', '.join(CLASS_NAMES)}",
        f" * Features: {len(FEATURE_NAMES)}",
        " */",
        "",
        "#include <math.h>",
        "",
    ]

    if scaler_path and os.path.exists(scaler_path):
        print(f"Incluyendo scaler: {scaler_path}")
        scaler = joblib.load(scaler_path)
        lines.append(generate_scaler_code(scaler))
        lines.append("")
        has_scaler = True
    else:
        lines.append(f"#define NUM_FEATURES {len(FEATURE_NAMES)}")
        lines.append("")
        has_scaler = False

    lines.append("// --- Modelo generado por m2cgen ---")
    lines.append(c_code)
    lines.append("")

    num_classes = len(CLASS_NAMES)
    if has_scaler:
        lines.append(generate_predict_wrapper(num_classes))

    lines.append(f"\n#endif // {header_guard}")

    with open(output_h, "w") as f:
        f.write("\n".join(lines))

    file_size = os.path.getsize(output_h)
    print(f"\nExportado: {output_h} ({file_size / 1024:.1f} KB)")
    print(f"Copiar al proyecto ESP32 y usar classify(features, &confidence)")

    return output_h


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python export_to_c.py <modelo.pkl> [scaler.pkl]")
        print("Ejemplo: python export_to_c.py decision_tree_best.pkl scaler_dt.pkl")
        sys.exit(1)

    model_path = sys.argv[1]
    scaler_path = sys.argv[2] if len(sys.argv) > 2 else None
    export_model(model_path, scaler_path)
