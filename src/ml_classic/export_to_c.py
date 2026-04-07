"""
=============================================================================
 export_to_c.py — Exporta modelos a código C puro para ESP32
=============================================================================
 Ahora soporta rutas absolutas/relativas en Linux y modelos entrenados con
 cuML: si el `.pkl` tiene `as_sklearn()`, primero se convierte a sklearn
 antes de pasarlo a `m2cgen`.
=============================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

try:
    from .training_runtime import ensure_output_dir, maybe_to_sklearn
except ImportError:
    from training_runtime import ensure_output_dir, maybe_to_sklearn


FEATURE_NAMES = [
    "num_pkts",
    "mean_iat",
    "std_iat",
    "min_iat",
    "max_iat",
    "mean_pkt_len",
    "num_bytes",
    "num_psh_flags",
    "num_rst_flags",
    "num_urg_flags",
    "std_pkt_len",
    "min_pkt_len",
    "max_pkt_len",
]

CLASS_NAMES = ["normal", "mqtt_bruteforce", "scan_A"]


def generate_scaler_code(scaler) -> str:
    means = scaler.mean_
    scales = scaler.scale_
    lines = [
        f"#define NUM_FEATURES {len(means)}",
        "",
        "static const float scaler_mean[NUM_FEATURES] = {",
        "    " + ", ".join(f"{value:.8f}f" for value in means),
        "};",
        "",
        "static const float scaler_scale[NUM_FEATURES] = {",
        "    " + ", ".join(f"{value:.8f}f" for value in scales),
        "};",
        "",
        "void apply_scaler(const float raw[NUM_FEATURES], float scaled[NUM_FEATURES]) {",
        "    for (int i = 0; i < NUM_FEATURES; i++) {",
        "        scaled[i] = (raw[i] - scaler_mean[i]) / scaler_scale[i];",
        "    }",
        "}",
    ]
    return "\n".join(lines)


def generate_predict_wrapper(num_classes: int, has_scaler: bool) -> str:
    classify_body = """
int classify(const float raw_features[NUM_FEATURES], float* confidence) {
"""
    if has_scaler:
        classify_body += """    float scaled[NUM_FEATURES];
    apply_scaler(raw_features, scaled);
    return predict_class(scaled, confidence);
}
"""
    else:
        classify_body += """    return predict_class(raw_features, confidence);
}
"""

    return f"""
#define NUM_CLASSES {num_classes}

int predict_class(const float features[NUM_FEATURES], float* confidence) {{
    double input[NUM_FEATURES];
    double scores[NUM_CLASSES];

    for (int i = 0; i < NUM_FEATURES; i++) {{
        input[i] = (double)features[i];
    }}

    score(input, scores);

    int best = 0;
    double best_score = scores[0];
    double sum_exp = 0.0;

    for (int i = 1; i < NUM_CLASSES; i++) {{
        if (scores[i] > best_score) {{
            best_score = scores[i];
            best = i;
        }}
    }}

    for (int i = 0; i < NUM_CLASSES; i++) {{
        sum_exp += exp(scores[i] - best_score);
    }}
    *confidence = (float)(1.0 / sum_exp);

    return best;
}}

{classify_body}
"""


def export_model(model_path: str, scaler_path: str | None = None, output_dir: str = ".", output_name: str | None = None):
    try:
        import m2cgen as m2c
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Falta `m2cgen`. Instálalo con `pip install m2cgen` antes de exportar a C."
        ) from exc

    model_path_obj = Path(model_path).expanduser().resolve()
    if not model_path_obj.exists():
        raise FileNotFoundError(f"No existe el modelo: {model_path_obj}")

    print(f"Cargando modelo: {model_path_obj}")
    model = joblib.load(model_path_obj)
    model = maybe_to_sklearn(model)
    model_name = output_name or f"{model_path_obj.stem}_esp32.h"

    print("Convirtiendo a código C con m2cgen...")
    c_code = m2c.export_to_c(model)

    output_dir_obj = ensure_output_dir(output_dir)
    output_h = output_dir_obj / model_name

    header_guard = output_h.stem.upper() + "_H"
    lines = [
        f"#ifndef {header_guard}",
        f"#define {header_guard}",
        "",
        "/*",
        f" * Auto-generated from {model_path_obj.name}",
        f" * Model: {type(model).__name__}",
        f" * Classes: {', '.join(CLASS_NAMES)}",
        f" * Features: {len(FEATURE_NAMES)}",
        " */",
        "",
        "#include <math.h>",
        "",
    ]

    has_scaler = False
    if scaler_path:
        scaler_path_obj = Path(scaler_path).expanduser().resolve()
        if not scaler_path_obj.exists():
            raise FileNotFoundError(f"No existe el scaler: {scaler_path_obj}")
        print(f"Incluyendo scaler: {scaler_path_obj}")
        scaler = joblib.load(scaler_path_obj)
        lines.append(generate_scaler_code(scaler))
        lines.append("")
        has_scaler = True
    else:
        lines.append(f"#define NUM_FEATURES {len(FEATURE_NAMES)}")
        lines.append("")

    lines.append("// --- Modelo generado por m2cgen ---")
    lines.append(c_code)
    lines.append("")
    lines.append(generate_predict_wrapper(len(CLASS_NAMES), has_scaler))
    lines.append(f"\n#endif // {header_guard}")

    output_h.write_text("\n".join(lines), encoding="utf-8")

    file_size = output_h.stat().st_size
    print(f"\nExportado: {output_h} ({file_size / 1024:.1f} KB)")
    print("Copiar al proyecto ESP32 y usar classify(features, &confidence)")
    return output_h


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exporta un modelo .pkl a header C para ESP32.")
    parser.add_argument("model_path", help="Ruta al archivo .pkl del modelo.")
    parser.add_argument("scaler_path", nargs="?", default=None, help="Ruta opcional al scaler .pkl.")
    parser.add_argument("--output-dir", default=".", help="Directorio donde se guardará el .h generado.")
    parser.add_argument(
        "--output-name",
        default=None,
        help="Nombre del archivo .h de salida. Si no se define, se usa <modelo>_esp32.h",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    export_model(
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        output_dir=args.output_dir,
        output_name=args.output_name,
    )
