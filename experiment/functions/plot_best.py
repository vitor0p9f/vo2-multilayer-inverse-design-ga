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
        plt.savefig(save_path, dpi=600)
    plt.show()


def _get_contrast_text_color(bg_color):
    """
    Return 'black' or 'white' depending on the luminance of the background color.
    Uses the standard relative luminance formula (sRGB coefficients).
    """
    r, g, b = bg_color[:3]  # ignore alpha if present
    # Calculate relative luminance
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return 'black' if luminance > 0.5 else 'white'


def _draw_structure(ax, structure: Structure, material_database: List[Material]) -> None:
    """
    Draw a vertical stack of only the visible layers (fixed or active free).
    Layers that are free and inactive are omitted.
    """
    # Determine which layers to show: fixed layers always visible,
    # free layers only if active_mask is True.
    free_mask = structure.free_thickness_mask | structure.free_material_mask
    show_mask = ~free_mask | structure.active_mask  # True = keep

    # Filter all layer attributes
    indices = structure.materials[show_mask]
    thicknesses = structure.thicknesses_m[show_mask]
    active_flags = structure.active_mask[show_mask]

    L_visible = len(indices)
    if L_visible == 0:
        ax.text(0.5, 0.5, 'No visible layers', ha='center', va='center',
                transform=ax.transAxes, fontsize=10)
        ax.axis('off')
        return

    # Symbol for each displayed layer
    symbols = [material_database[int(idx)].symbol for idx in indices]

    # --- Fixed colour mapping by material symbol ---
    # Gather all unique symbols from the entire material database, preserving order
    all_symbols = [mat.symbol for mat in material_database]
    # Remove duplicates while keeping order of first occurrence
    unique_symbols = list(dict.fromkeys(all_symbols))
    # Use a perceptually uniform colormap, mapping each symbol to a fixed colour
    cmap = plt.cm.tab20  # up to 20 distinct colours
    n_unique = len(unique_symbols)
    # For more than 20 materials, cycle the colormap with a slight offset
    colour_dict = {}
    for i, sym in enumerate(unique_symbols):
        if n_unique <= 20:
            colour_dict[sym] = cmap(i / max(1, n_unique - 1))
        else:
            # For >20, use a larger cycler (e.g., tab20 + tab20b) or just modulo
            colour_dict[sym] = cmap(i % 20)

    # Labels: material symbol + thickness (or "semi‑infinite")
    labels = []
    for i in range(L_visible):
        d = float(thicknesses[i])
        if jnp.isinf(d):
            lbl = f"{symbols[i]}\nsemi‑infinite"
        else:
            lbl = f"{symbols[i]}\n{d * 1e9:.1f} nm"
        labels.append(lbl)

    # Visual parameters
    bar_width = 0.6
    layer_height = 1.5
    y_top = L_visible * layer_height

    for lbl, sym in zip(labels, symbols):
        y_bottom = y_top - layer_height
        colour = colour_dict[sym]
        text_color = _get_contrast_text_color(colour)  # black or white for best visibility
        ax.bar(0, layer_height, bottom=y_bottom, width=bar_width,
               color=colour, edgecolor='k', linewidth=1.2)
        ax.text(0, y_bottom + layer_height / 2, lbl,
                ha='center', va='center', fontsize=8, fontweight='bold',
                color=text_color)
        y_top = y_bottom

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0, L_visible * layer_height)
    ax.axis('off')
    ax.set_title("Structure layers", fontsize=12, fontweight='normal')