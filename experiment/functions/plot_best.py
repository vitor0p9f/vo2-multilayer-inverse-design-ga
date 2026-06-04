from typing import List, Optional, Union
import matplotlib.pyplot as plt
import jax.numpy as jnp
from pathlib import Path

from specifications.objects.structure import Structure
from specifications.objects.material import Material
from specifications.types.environment import Environment
from specifications.types.optics import Reflectance, Transmittance, Absorptance
from genetic_algorithm.types.evaluation_result import EvaluationResult


def plot_best_structure(
    result: EvaluationResult,
    env: Environment,
    target: Union[Reflectance, Transmittance, Absorptance],
    material_database: List[Material],
    save_path: Optional[Path] = None,
    generation: Optional[int] = None,
) -> None:
    """
    Plot the best structure and its optical response vs. the target.

    Left panel: vertical stack of layers with material symbol and thickness.
    Right panel: non‑polarised computed spectra for all temperatures (dashed)
                 and the target spectrum (solid).
    """
    # ------------------------------------------------------------------
    # 1.  Prepare data
    # ------------------------------------------------------------------
    struct = result.structure
    wavelengths_nm = env.wavelengths * 1e9   # (W,) in nanometres

    # Determine which property to plot based on the target's type
    if isinstance(target, Reflectance):
        computed = result.reflectance       # (T, A, W)
        target_name = "Reflectance"
    elif isinstance(target, Transmittance):
        computed = result.transmittance
        target_name = "Transmittance"
    else:
        computed = result.absorptance
        target_name = "Absorptance"

    target_np = target.non_polarized        # (1, W) or (1, 1, W)
    target_np = jnp.squeeze(target_np)      # (W,)

    T = computed.shape[0]   # number of temperatures
    # Use the first angle (index 0) for simplicity
    computed_single_angle = computed[:, 0, :]  # (T, W)

    # ------------------------------------------------------------------
    # 2.  Create figure
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2])

    # --- Left panel: structure stack ---
    ax_struct = fig.add_subplot(gs[0, 0])
    _draw_structure(ax_struct, struct, material_database)

    # Use the right spine of the structure panel as a divider
    ax_struct.spines['right'].set_visible(True)
    ax_struct.spines['right'].set_linestyle(':')
    ax_struct.spines['right'].set_linewidth(0.8)
    ax_struct.spines['right'].set_color('grey')

    # --- Right panel: spectra ---
    ax_spec = fig.add_subplot(gs[0, 1])
    # Hide the left spine of the spectrum panel to avoid a double line
    ax_spec.spines['left'].set_visible(False)

    # Plot computed for each temperature (dashed)
    for t in range(T):
        temp_label = f"{env.temperatures[t]:.0f} K"
        ax_spec.plot(wavelengths_nm, computed_single_angle[t, :],
                     linestyle='--', alpha=0.8, label=temp_label)
    # Plot target (solid, thicker)
    ax_spec.plot(wavelengths_nm, target_np, 'k-', linewidth=2, label='Target')

    ax_spec.set_xlabel("Wavelength (nm)")
    ax_spec.set_ylabel(target_name)
    # Cost with 6 decimal places
    ax_spec.set_title(f"Best structure – cost = {result.cost:.6f}")
    ax_spec.grid(True, alpha=0.3)
    ax_spec.legend(fontsize='small', loc='best')

    # ---- Main title ----
    if generation is not None:
        fig.suptitle(f"Generation {generation}", fontsize=14, fontweight='bold', y=0.98)
    else:
        fig.suptitle("Best structure – Optical response", fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])   # leave room for suptitle
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def _draw_structure(ax, structure: Structure, material_database: List[Material]) -> None:
    """
    Draw a vertical stack of layers with material‑consistent colours and
    labels centred inside each rectangle.
    """
    L = structure.materials.shape[0]
    indices = structure.materials
    thicknesses = structure.thicknesses_m

    # Symbol for each layer
    symbols = [material_database[int(idx)].symbol for idx in indices]

    # Fixed colour per distinct material
    unique_mats = list(set(indices.tolist()))
    cmap = plt.cm.tab10
    colour_dict = {mat: cmap(i % 10) for i, mat in enumerate(unique_mats)}

    # Labels: material symbol + thickness (or "semi‑infinite")
    labels = []
    for i in range(L):
        d = float(thicknesses[i])
        if jnp.isinf(d):
            lbl = f"{symbols[i]}\nsemi‑infinite"
        else:
            lbl = f"{symbols[i]}\n{d * 1e9:.1f} nm"
        labels.append(lbl)

    # Visual parameters
    bar_width = 0.6                # narrower for cleaner centering
    layer_height = 1.5
    y_top = L * layer_height

    for lbl, idx in zip(labels, indices):
        y_bottom = y_top - layer_height
        colour = colour_dict[int(idx)]
        ax.bar(0, layer_height, bottom=y_bottom, width=bar_width,
               color=colour, edgecolor='k', linewidth=1.2)
        ax.text(0, y_bottom + layer_height / 2, lbl,
                ha='center', va='center', fontsize=8, fontweight='bold',
                color='white')
        y_top = y_bottom

    # Symmetric x‑limits to perfectly centre the stack
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0, L * layer_height)
    ax.axis('off')
    ax.set_title("Structure layers", fontsize=12, fontweight='normal')