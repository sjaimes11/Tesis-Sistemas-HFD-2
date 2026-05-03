"""
Genera el diagrama de secuencia bidireccional HFL v7 que reemplaza el flujo
horizontal en linea de la Seccion 3.6 del documento.

Inspirado en mermaid sequence diagrams (3 fases). Salida:
    hfl_v7_bidirectional_flow.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
ACTORS = [
    ("PC\n(Cloud Coordinator)", "#1F3A6E"),
    ("Raspberry Pi\n(Fog Gateway)", "#1F6E4A"),
    ("ESP32-S3\n(Edge Node)", "#A14A1F"),
]

# Coordenadas X de las lifelines
ACTOR_X = {name: 1.5 + 3.0 * i for i, (name, _) in enumerate(ACTORS)}

PHASE_BG = {
    "PHASE 1 - Top-Down Model Deployment": "#E8EFF7",
    "PHASE 2 - Local Data Collection & Training": "#E9F4EB",
    "PHASE 3 - Bottom-Up Aggregation (FedAvg)": "#F7EFE8",
}


def actor_box(ax, x: float, y: float, label: str, color: str) -> None:
    box = mpatches.FancyBboxPatch(
        (x - 0.85, y - 0.32),
        1.7, 0.64,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.4,
        edgecolor=color,
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center",
            color="white", fontsize=9.5, fontweight="bold")


def phase_band(ax, y: float, label: str, color: str, width_total: float = 9.5) -> None:
    band = mpatches.Rectangle(
        (0.4, y - 0.22),
        width_total, 0.44,
        linewidth=0,
        facecolor=color,
        alpha=0.95,
    )
    ax.add_patch(band)
    ax.text(0.4 + width_total / 2, y, label, ha="center", va="center",
            color="#1A1A1A", fontsize=10.5, fontweight="bold")


def message_arrow(ax, x_from: float, x_to: float, y: float,
                  label: str, color: str = "#1A1A1A",
                  style: str = "->", lw: float = 1.6) -> None:
    """Mensaje horizontal entre dos actores. La etiqueta se coloca encima de
    la flecha (con eje Y invertido, eso es y - offset) y con fondo blanco
    para no superponerse a la linea."""
    ax.annotate(
        "",
        xy=(x_to, y), xycoords="data",
        xytext=(x_from, y), textcoords="data",
        arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                        shrinkA=0, shrinkB=0),
        zorder=2,
    )
    midx = (x_from + x_to) / 2
    ax.text(
        midx, y - 0.18, label,
        ha="center", va="bottom",
        fontsize=9, color=color, zorder=3,
        bbox=dict(facecolor="white", edgecolor="none", pad=2.5, alpha=0.95),
    )


def self_loop(ax, x: float, y: float, label: str,
              color: str = "#1A1A1A", side: str = "right") -> None:
    """Auto-mensaje: pequeno bucle rectangular pegado al actor con la
    etiqueta inmediatamente al lado de la curva.

    side: 'right' o 'left'. Determina hacia donde se dibuja el bucle y donde
    se coloca el texto. Util para PC (left), ESP32 (right), Pi (cualquiera).
    """
    direction = 1 if side == "right" else -1
    loop_w = 0.32
    loop_h = 0.30
    half_h = loop_h / 2

    # Tres segmentos del bucle: arriba sale a la derecha/izquierda,
    # baja, y vuelve con flecha.
    ax.plot([x, x + loop_w * direction],
            [y - half_h, y - half_h],
            color=color, lw=1.3, zorder=2)
    ax.plot([x + loop_w * direction, x + loop_w * direction],
            [y - half_h, y + half_h],
            color=color, lw=1.3, zorder=2)
    ax.annotate(
        "",
        xy=(x, y + half_h),
        xytext=(x + loop_w * direction, y + half_h),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.3,
                        shrinkA=0, shrinkB=0),
        zorder=2,
    )

    # Etiqueta inmediatamente al lado del bucle.
    text_x = x + (loop_w + 0.12) * direction
    text_ha = "left" if side == "right" else "right"
    ax.text(
        text_x, y, label,
        ha=text_ha, va="center",
        fontsize=8.7, color=color, style="italic",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.95),
        zorder=3,
    )


def main(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 9.5))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 13.5)
    ax.invert_yaxis()
    ax.set_axis_off()

    # Encabezado
    ax.text(5.25, 0.5,
            "Flujo bidireccional HFL v7 (Edge -- Fog -- Cloud)",
            ha="center", va="center", fontsize=12, fontweight="bold")

    # Cabecera con actores
    actor_y_top = 1.4
    for (name, color) in ACTORS:
        actor_box(ax, ACTOR_X[name], actor_y_top, name, color)

    # Lifelines (verticales discontinuas)
    lifeline_top = actor_y_top + 0.32
    lifeline_bottom = 13.0
    for (name, _) in ACTORS:
        ax.plot([ACTOR_X[name], ACTOR_X[name]],
                [lifeline_top, lifeline_bottom],
                linestyle="--", color="#888888", linewidth=0.9, zorder=1)

    pc = ACTORS[0][0]; pi = ACTORS[1][0]; edge = ACTORS[2][0]

    # ----------------- PHASE 1 -----------------
    y = 2.6
    phase_band(ax, y, "PHASE 1  -  Top-Down Model Deployment", PHASE_BG["PHASE 1 - Top-Down Model Deployment"])

    y += 0.7
    message_arrow(ax, ACTOR_X[pc], ACTOR_X[pi], y,
                  "HTTP POST /deploy-model  (pesos globales)",
                  color="#1F3A6E")

    y += 0.7
    self_loop(ax, ACTOR_X[pi], y,
              "Validar tag ASCON + actualizar modelo local",
              color="#1F6E4A")

    y += 0.7
    message_arrow(ax, ACTOR_X[pi], ACTOR_X[edge], y,
                  "MQTT publish fl/global_model  (envelope ASCON)",
                  color="#1F6E4A")

    y += 0.7
    self_loop(ax, ACTOR_X[edge], y,
              "Descifrar + sobrescribir pesos federados",
              color="#A14A1F")

    # ----------------- PHASE 2 -----------------
    y += 1.0
    phase_band(ax, y, "PHASE 2  -  Local Data Collection & Training", PHASE_BG["PHASE 2 - Local Data Collection & Training"])

    y += 0.7
    message_arrow(ax, ACTOR_X[edge], ACTOR_X[pi], y,
                  "MQTT publish fl/features  (loop cada ~5 s, ASCON)",
                  color="#A14A1F")

    y += 0.7
    self_loop(ax, ACTOR_X[edge], y,
              "Inferencia TinyML + alerting local",
              color="#A14A1F")

    y += 0.7
    self_loop(ax, ACTOR_X[pi], y,
              "Etiquetado heuristico + buffer de N_s = 40 muestras",
              color="#1F6E4A")

    y += 0.7
    self_loop(ax, ACTOR_X[pi], y,
              "Entrenamiento local: 5 epocas, batch 8, Adam (lr=0.005)",
              color="#1F6E4A")

    # ----------------- PHASE 3 -----------------
    y += 1.0
    phase_band(ax, y, "PHASE 3  -  Bottom-Up Aggregation (FedAvg)", PHASE_BG["PHASE 3 - Bottom-Up Aggregation (FedAvg)"])

    y += 0.7
    message_arrow(ax, ACTOR_X[pi], ACTOR_X[pc], y,
                  "HTTP POST /aggregate-from-gateway  (pesos + n_k)",
                  color="#1F6E4A")

    y += 0.7
    self_loop(ax, ACTOR_X[pc], y,
              "Esperar K = 2 gateways + FedAvg ponderado por muestras",
              color="#1F3A6E", side="right")

    y += 0.7
    self_loop(ax, ACTOR_X[pc], y,
              "Actualizar modelo global  ->  nueva ronda",
              color="#1F3A6E", side="right")

    # Pie de pagina con notacion
    ax.text(5.25, 13.3,
            "ASCON-128 sobre todos los canales  |  En modo FOG: + fog/weights y fog/global_model entre RPi peer/leader",
            ha="center", va="center", fontsize=8.5, color="#444444",
            style="italic")

    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"OK -> {output}")


if __name__ == "__main__":
    out = Path(__file__).parent / "hfl_v7_bidirectional_flow.png"
    main(out)
