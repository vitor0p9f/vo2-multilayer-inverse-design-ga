# experiment/functions/generate_report.py

from pathlib import Path
from typing import List, Union, Optional
import json
import matplotlib.pyplot as plt
import jax.numpy as jnp

from specifications.objects.structure import Structure
from specifications.objects.material import Material
from specifications.types.environment import Environment
from specifications.types.optics import Reflectance, Transmittance, Absorptance
from genetic_algorithm.types.evaluation_result import EvaluationResult
from genetic_algorithm.functions.evaluate_population import evaluate_population

# Reuse the drawing helpers from the existing plot module
from experiment.functions.plot_best import _draw_structure, _get_contrast_text_color


# ----------------------------------------------------------------------
#  Plot: structure + R, T, A at one temperature
# ----------------------------------------------------------------------
def plot_structure_and_spectra(
    result: EvaluationResult,
    env: Environment,
    material_database: List[Material],
    temperature_index: int = 0,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot the structure (left) and unpolarised R, T, A at the chosen temperature (right).
    """
    struct = result.structure
    wavelengths_nm = env.wavelengths * 1e9

    # Extract spectra for the first angle (0)
    R = result.reflectance[:, 0, :]       # (T, W)
    T = result.transmittance[:, 0, :]
    A = result.absorptance[:, 0, :]

    temp = env.temperatures[temperature_index]
    r_curve = R[temperature_index, :]
    t_curve = T[temperature_index, :]
    a_curve = A[temperature_index, :]

    fig = plt.figure(figsize=(12, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2])

    # Left panel: structure stack
    ax_struct = fig.add_subplot(gs[0, 0])
    _draw_structure(ax_struct, struct, material_database)   # <-- reused
    ax_struct.spines['right'].set_visible(True)
    ax_struct.spines['right'].set_linestyle(':')
    ax_struct.spines['right'].set_linewidth(0.8)
    ax_struct.spines['right'].set_color('grey')

    # Right panel: R, T, A
    ax_spec = fig.add_subplot(gs[0, 1])
    ax_spec.spines['left'].set_visible(False)

    ax_spec.plot(wavelengths_nm, r_curve, label='Reflectance', color='C0')
    ax_spec.plot(wavelengths_nm, t_curve, label='Transmittance', color='C2')
    ax_spec.plot(wavelengths_nm, a_curve, label='Absorptance', color='C3')

    ax_spec.set_ylim(0.0, 1.0)

    ax_spec.set_xlabel("Wavelength (nm)")
    ax_spec.set_ylabel("Value")
    ax_spec.set_title(f"Optical response at {temp:.0f} K — cost = {result.cost:.6f}")
    ax_spec.grid(True, alpha=0.3)
    ax_spec.legend(fontsize='small', loc='best')

    fig.suptitle("Best structure – Full optical response", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=600)
    plt.show()


# ----------------------------------------------------------------------
#  Report helpers
# ----------------------------------------------------------------------
def _plot_convergence(directory: Path) -> None:
    """Plot best cost per generation from cost_history.json."""
    history_file = directory / "cost_history.json"
    if not history_file.exists():
        print("No cost_history.json found. Skipping convergence plot.")
        return

    with open(history_file) as f:
        history = json.load(f)

    plt.figure(figsize=(8, 4))
    plt.plot(range(len(history)), history, 'o-', markersize=3)
    plt.xlabel("Generation")
    plt.ylabel("Best cost")
    plt.title("Convergence history")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _load_best_structure_from_file(directory: Path) -> Optional[Structure]:
    """
    Load the single best structure from 'elite_final.json'.
    The file may contain a single structure object or a list;
    if it's a list, the first element is used.

    Since the JSON does not contain `free_thickness_mask` or `free_material_mask`,
    we treat all layers as "free" so that `active_mask` alone determines visibility.
    """
    # Primary: elite_final.json
    file_path = directory / "elite_final.json"
    if not file_path.exists():
        # Fallback: best.json or best5.json
        for fallback in ("best.json", "best5.json"):
            fallback_path = directory / fallback
            if fallback_path.exists():
                file_path = fallback_path
                break
        else:
            return None

    with open(file_path) as f:
        data = json.load(f)

    # Handle both a single object and a list
    if isinstance(data, list):
        entry = data[0]   # the best is always first
    else:
        entry = data

    # Reconstruct Structure
    active_mask = jnp.array(entry["active_mask"], dtype=bool)
    L = len(active_mask)

    # Force all layers to be considered "free" so that only active_mask
    # controls visibility in _draw_structure. This is necessary because
    # the JSON does not contain the original free masks.
    free_all = jnp.ones(L, dtype=bool)   # all True

    return Structure(
        materials=jnp.array(entry["materials"], dtype=jnp.int8),
        thicknesses_m=jnp.array(entry["thicknesses_m"], dtype=jnp.float64),
        active_mask=active_mask,
        free_thickness_mask=free_all,
        free_material_mask=free_all,
    )


# ----------------------------------------------------------------------
#  Main report function
# ----------------------------------------------------------------------
def generate_experiment_report(
    experiment_dir: Path,
    env: Environment,
    target: Union[Reflectance, Transmittance, Absorptance],
    materials: List[Material],
    temperature_index: int = 0,
) -> None:
    """
    Generate full experiment report:
      - Convergence history
      - The single best structure with layer stack and R,T,A at the given temperature.
    """
    print("=== Experiment Report ===")
    print(f"Directory: {experiment_dir}")

    # Convergence
    _plot_convergence(experiment_dir)

    # Load best structure
    best_struct = _load_best_structure_from_file(experiment_dir)
    if best_struct is None:
        print("No best structure JSON file found. Skipping structure plot.")
        return

    # Wrap it in a population of size 1 and evaluate to get spectra
    from genetic_algorithm.functions.convert_structures_to_population import structures_to_population
    pop = structures_to_population([best_struct])
    # evaluate_population returns a list of EvaluationResults
    results = evaluate_population(env, pop, materials, target, ())
    if not results:
        print("Evaluation of best structure produced no results.")
        return

    result = results[0]

    # Obtain the historical cost from the JSON file
    historical_cost = 0.0
    for filename in ("elite_final.json", "best.json", "best5.json"):
        file_path = experiment_dir / filename
        if file_path.exists():
            with open(file_path) as f:
                data = json.load(f)
            entry = data[0] if isinstance(data, list) else data
            historical_cost = entry.get("cost", 0.0)
            break

    result_with_cost = EvaluationResult(
        structure=result.structure,
        cost=historical_cost,
        reflectance=result.reflectance,
        transmittance=result.transmittance,
        absorptance=result.absorptance,
    )

    print(f"Best structure cost = {historical_cost:.6f}")
    plot_structure_and_spectra(
        result=result_with_cost,
        env=env,
        material_database=materials,
        temperature_index=temperature_index,
    )