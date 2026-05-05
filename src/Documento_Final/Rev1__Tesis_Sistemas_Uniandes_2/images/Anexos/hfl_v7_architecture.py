"""
Diagrama de arquitectura del sistema HFL v7 (Edge - Fog - Cloud).

Reemplaza la descripcion narrativa del Anexo A por una figura que muestra:
- Las tres capas (Edge, Fog, Cloud) con su hardware tipico.
- Los protocolos por enlace (HTTP, MQTT) y la proteccion ASCON-128.
- La extension FOG con un par leader/peer sobre Raspberry Pi.

Salida: hfl_v7_architecture.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paleta consistente con el diagrama de secuencia
# ---------------------------------------------------------------------------
COLOR_CLOUD = "#1F3A6E"   # azul navy
COLOR_FOG   = "#1F6E4A"   # verde fog
COLOR_EDGE  = "#A14A1F"   # naranja edge
ASCON_COLOR = "#7E1E1E"

LAYER_BG = {
    "cloud": "#EAF1FA",
    "fog":   "#EAF6EE",
    "edge":  "#FCEFE5",
}


# ---------------------------------------------------------------------------
# Componentes de dibujo
# ---------------------------------------------------------------------------

def device_box(ax, x: float, y: float, w: float, h: float,
               title: str, subtitle: str, color: str) -> None:
    """Dibuja una caja de dispositivo con banda de titulo en la parte superior
    y subtitulo abajo. Convencion: y crece hacia arriba.
    """
    band_height = 0.36
    # Cuerpo (blanco con borde de color)
    body = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w, h,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=1.6,
        edgecolor=color,
        facecolor="white",
        zorder=4,
    )
    ax.add_patch(body)
    # Banda superior coloreada
    band = mpatches.Rectangle(
        (x - w / 2 + 0.04, y + h / 2 - band_height),
        w - 0.08, band_height - 0.04,
        linewidth=0,
        facecolor=color,
        zorder=5,
    )
    ax.add_patch(band)
    ax.text(x, y + h / 2 - band_height / 2 - 0.02, title,
            ha="center", va="center",
            color="white", fontsize=10, fontweight="bold", zorder=6)
    # Subtitulo bajo la banda
    ax.text(x, y - h / 2 + (h - band_height) / 2 - 0.05, subtitle,
            ha="center", va="center",
            color="#222222", fontsize=8.5, zorder=6,
            linespacing=1.35)


def layer_band(ax, y_top: float, y_bottom: float, color_bg: str,
               label: str, label_color: str,
               x_left: float = 0.30, x_right: float = 13.70) -> None:
    """Banda horizontal de fondo + etiqueta vertical de capa."""
    band = mpatches.FancyBboxPatch(
        (x_left, y_bottom),
        x_right - x_left, y_top - y_bottom,
        boxstyle="round,pad=0,rounding_size=0.10",
        linewidth=0,
        facecolor=color_bg,
        zorder=1,
    )
    ax.add_patch(band)
    ax.text(x_left + 0.18, (y_top + y_bottom) / 2, label,
            ha="left", va="center",
            color=label_color, fontsize=11, fontweight="bold",
            rotation=90, zorder=2)


def link_arrow(ax, x1: float, y1: float, x2: float, y2: float,
               color: str = "#333333", lw: float = 1.6,
               style: str = "<->",
               label_top: str | None = None,
               label_bottom: str | None = None,
               label_dy: float = 0.18) -> None:
    """Flecha bidireccional con etiquetas opcionales (arriba/abajo del medio)."""
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                        shrinkA=4, shrinkB=4),
        zorder=3,
    )
    midx = (x1 + x2) / 2
    midy = (y1 + y2) / 2
    if label_top:
        ax.text(midx, midy + label_dy, label_top,
                ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=color,
                bbox=dict(facecolor="white", edgecolor="none", pad=2.5),
                zorder=4)
    if label_bottom:
        ax.text(midx, midy - label_dy, label_bottom,
                ha="center", va="top",
                fontsize=8, color=ASCON_COLOR, style="italic",
                bbox=dict(facecolor="white", edgecolor="none", pad=2.5),
                zorder=4)


def fog_link(ax, x1: float, y: float, x2: float) -> None:
    """Enlace MQTT especifico entre Raspberry Pi peer y leader (linea
    discontinua + etiqueta sobre la linea)."""
    ax.annotate(
        "",
        xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(arrowstyle="<->", color=COLOR_FOG, lw=1.6,
                        linestyle=(0, (5, 3)),
                        shrinkA=4, shrinkB=4),
        zorder=3,
    )
    midx = (x1 + x2) / 2
    ax.text(midx, y + 0.20,
            "MQTT  fog/weights  +  fog/global_model",
            ha="center", va="bottom",
            fontsize=8.8, fontweight="bold", color=COLOR_FOG,
            bbox=dict(facecolor="white", edgecolor="none", pad=2.5),
            zorder=4)
    ax.text(midx, y - 0.20,
            "(solo en variantes *_FOG)",
            ha="center", va="top",
            fontsize=7.8, style="italic", color="#555555",
            bbox=dict(facecolor="white", edgecolor="none", pad=2.5),
            zorder=4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(output: Path) -> None:
    # Lienzo mas ancho para que las etiquetas no se trunquen y los 4 ESP32
    # tengan separacion suficiente.
    fig, ax = plt.subplots(figsize=(16.5, 10.0))
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 11.5)
    ax.set_axis_off()
    # Eje normal: y=11 arriba, y=0 abajo. Cloud arriba, Edge abajo.

    # --- Encabezado ---
    ax.text(8.25, 11.0,
            "Arquitectura del Sistema HFL v7  -  Edge / Fog / Cloud",
            ha="center", va="center", fontsize=14, fontweight="bold")

    # --- Bandas de capa ---
    layer_band(ax, y_top=10.55, y_bottom=8.95, color_bg=LAYER_BG["cloud"],
               label="CAPA CLOUD", label_color=COLOR_CLOUD,
               x_left=0.30, x_right=16.20)
    layer_band(ax, y_top=7.40,  y_bottom=4.65, color_bg=LAYER_BG["fog"],
               label="CAPA FOG",   label_color=COLOR_FOG,
               x_left=0.30, x_right=16.20)
    layer_band(ax, y_top=3.10,  y_bottom=0.40, color_bg=LAYER_BG["edge"],
               label="CAPA EDGE",  label_color=COLOR_EDGE,
               x_left=0.30, x_right=16.20)

    # --- Capa Cloud: PC ---
    pc_x, pc_y = 8.25, 9.75
    device_box(ax, x=pc_x, y=pc_y, w=5.4, h=1.30,
               title="PC  -  Cloud Coordinator",
               subtitle="server_hfl.py  /  server_hfl_fog.py\n"
                        "FastAPI + FedAvg ponderado + dashboard analitico",
               color=COLOR_CLOUD)

    # --- Capa Fog: Raspberry Pi leader y peer ---
    pi_y = 5.95
    pi_leader_x = 4.20
    pi_peer_x   = 12.30
    pi_w, pi_h  = 4.80, 1.50
    device_box(ax, x=pi_leader_x, y=pi_y, w=pi_w, h=pi_h,
               title="Raspberry Pi  -  Fog Gateway",
               subtitle="gateway_hfl.py  /  gateway_hfl_fog.py\n"
                        "Mosquitto broker  +  etiquetado heuristico\n"
                        "Buffer N=40  +  entrenamiento local Keras",
               color=COLOR_FOG)
    device_box(ax, x=pi_peer_x, y=pi_y, w=pi_w, h=pi_h,
               title="Raspberry Pi  -  FOG peer (opcional)",
               subtitle="gateway_hfl_fog.py  (FOG_ROLE=peer)\n"
                        "Mismo broker / etiquetado / entrenamiento\n"
                        "Reporta pesos al leader (solo modo FOG)",
               color=COLOR_FOG)

    # --- Capa Edge: 4 ESP32 (2 por Pi) ---
    edge_y      = 1.65
    edge_w      = 2.30
    edge_h      = 1.55
    cluster_dx  = 1.45  # separacion del centro del Pi a cada ESP32

    # Cluster bajo el leader
    device_box(ax, x=pi_leader_x - cluster_dx, y=edge_y, w=edge_w, h=edge_h,
               title="ESP32-S3  A.normal",
               subtitle="main_edge_node_normal.cpp\n"
                        "TinyML inference\n13 features de flujo\n"
                        "100% trafico normal",
               color=COLOR_EDGE)
    device_box(ax, x=pi_leader_x + cluster_dx, y=edge_y, w=edge_w, h=edge_h,
               title="ESP32-S3  A.simulated",
               subtitle="main_edge_node_simulated.cpp\n"
                        "TinyML inference\n13 features de flujo\n"
                        "40% normal / 30% brute / 30% scan",
               color=COLOR_EDGE)
    # Cluster bajo el peer
    device_box(ax, x=pi_peer_x - cluster_dx, y=edge_y, w=edge_w, h=edge_h,
               title="ESP32-S3  B.normal",
               subtitle="main_edge_node_normal.cpp\n"
                        "TinyML inference\n13 features de flujo\n"
                        "100% trafico normal",
               color=COLOR_EDGE)
    device_box(ax, x=pi_peer_x + cluster_dx, y=edge_y, w=edge_w, h=edge_h,
               title="ESP32-S3  B.simulated",
               subtitle="main_edge_node_simulated.cpp\n"
                        "TinyML inference\n13 features de flujo\n"
                        "40% normal / 30% brute / 30% scan",
               color=COLOR_EDGE)

    # --- Enlaces PC <-> RPi (HTTP + ASCON) ---
    pc_bottom = pc_y - 0.65
    pi_top    = pi_y + 0.75
    link_arrow(ax,
               x1=pc_x - 1.8, y1=pc_bottom,
               x2=pi_leader_x + 1.0, y2=pi_top,
               color=COLOR_CLOUD,
               label_top="HTTP  /deploy-model  +  /aggregate-from-gateway*",
               label_bottom="ASCON-128  (envelope JSON)",
               label_dy=0.34)
    link_arrow(ax,
               x1=pc_x + 1.8, y1=pc_bottom,
               x2=pi_peer_x - 1.0, y2=pi_top,
               color=COLOR_CLOUD,
               label_top="HTTP  /deploy-model  +  /aggregate-from-fog",
               label_bottom="ASCON-128  (envelope JSON)",
               label_dy=0.34)

    # --- Enlace fog (Pi peer <-> Pi leader) ---
    # Va POR ENCIMA de los Pi para no chocar con sus subtitulos.
    fog_y = pi_y + pi_h / 2 + 0.42
    fog_link(ax,
             x1=pi_leader_x + 1.20, y=fog_y,
             x2=pi_peer_x - 1.20)

    # --- Enlaces RPi <-> ESP32 (MQTT + ASCON) ---
    pi_bottom = pi_y - 0.75
    edge_top  = edge_y + edge_h / 2
    for (px, ex) in [
        (pi_leader_x, pi_leader_x - cluster_dx),
        (pi_leader_x, pi_leader_x + cluster_dx),
        (pi_peer_x,   pi_peer_x - cluster_dx),
        (pi_peer_x,   pi_peer_x + cluster_dx),
    ]:
        link_arrow(ax,
                   x1=px, y1=pi_bottom,
                   x2=ex, y2=edge_top,
                   color=COLOR_EDGE,
                   style="<->",
                   lw=1.4)

    # Etiqueta unica para cada cluster MQTT, situada justo encima del
    # vertice de las flechas para no caer dentro de las cajas Pi/ESP32.
    cluster_label_y = pi_bottom - 0.30
    for cluster_x in (pi_leader_x, pi_peer_x):
        ax.text(cluster_x, cluster_label_y, "MQTT  fl/features  /  fl/global_model",
                ha="center", va="top",
                fontsize=9, fontweight="bold", color=COLOR_EDGE,
                bbox=dict(facecolor="white", edgecolor="none", pad=2.5),
                zorder=5)
        ax.text(cluster_x, cluster_label_y - 0.32, "ASCON-128  (envelope JSON)",
                ha="center", va="top",
                fontsize=8, style="italic", color=ASCON_COLOR,
                bbox=dict(facecolor="white", edgecolor="none", pad=2.5),
                zorder=5)

    # --- Pie de figura con notas ---
    note = (
        "ASCON-128 protege todos los canales en variantes *_ASCON_*  |  "
        "En variantes *_PLAIN_* los topics y endpoints usan sufijo  _plain  (JSON sin cifrar)  |  "
        "El modo FOG activa el enlace y el peer marcados con linea discontinua"
    )
    ax.text(8.25, 0.10, note,
            ha="center", va="bottom",
            fontsize=9, color="#444444", style="italic")

    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"OK -> {output}")


if __name__ == "__main__":
    out = Path(__file__).parent / "hfl_v7_architecture.png"
    main(out)
