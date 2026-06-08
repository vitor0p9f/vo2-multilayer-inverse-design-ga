from pathlib import Path
from typing import List, Union
import json
import matplotlib.pyplot as plt
import jax.numpy as jnp

from specifications.objects.structure import Structure
from specifications.objects.material import Material
from specifications.types.environment import Environment
from specifications.types.optics import Reflectance, Transmittance, Absorptance, SpectralBand
from genetic_algorithm.types.evaluation_result import EvaluationResult
from genetic_algorithm.functions.evaluate_population import evaluate_population
from genetic_algorithm.objects.population import Population
from experiment.functions.plot_best import plot_best_structure


def generate_experiment_report(
    experiment_dir: Path,
    env: Environment,
    target: Union[Reflectance, Transmittance, Absorptance],
    materials: List[Material],
) -> None:
    """
    Generate a full report of the finished GA experiment.

    Displays:
      - Convergence history (best cost vs. generation)
      - For each of the top‑5 structures: layer stack + computed spectra vs target.
    """
    print("=== Experiment Report ===")
    print(f"Directory: {experiment_dir}")

    # 1. Convergence plot
    _plot_convergence(experiment_dir)

    # 2. Best structures (re‑evaluated so we have spectra)
    _plot_best_structures(experiment_dir, env, target, materials)


def _plot_convergence(directory: Path) -> None:
    """Plot best cost per generation from cost_history.json."""
    history_file = directory / "cost_history.json"
    if not history_file.exists():
        print("No cost_history.json found. Skipping convergence plot.")
        return

    with open(history_file) as f:
        history = json.load(f)

    gens = list(range(len(history)))
    plt.figure(figsize=(8, 4))
    plt.plot(gens, history, 'o-', markersize=3)
    plt.xlabel("Generation")
    plt.ylabel("Best cost")
    plt.title("Convergence history")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _plot_best_structures(
    directory: Path,
    env: Environment,
    target: Union[Reflectance, Transmittance, Absorptance],
    materials: List[Material],
) -> None:
    """Re‑evaluate the top‑5 structures and plot them with full optical response."""
    best5_file = directory / "best5.json"
    if not best5_file.exists():
        print("No best5.json found. Skipping structure plots.")
        return

    with open(best5_file) as f:
        entries = json.load(f)

    # Reconstruct structures and their historical costs
    structs = []
    hist_costs = []
    for entry in entries:
        # Convert list to JAX array to avoid zeros_like error
        active_mask_arr = jnp.array(entry["active_mask"], dtype=bool)
        s = Structure(
            materials=jnp.array(entry["materials"], dtype=jnp.int8),
            thicknesses_m=jnp.array(entry["thicknesses_m"], dtype=jnp.float64),
            active_mask=active_mask_arr,
            free_thickness_mask=jnp.zeros_like(active_mask_arr, dtype=bool),  # now works
            free_material_mask=jnp.zeros_like(active_mask_arr, dtype=bool),
        )
        structs.append(s)
        hist_costs.append(entry["cost"])

    # Build a temporary Population of size 5 and evaluate it
    from genetic_algorithm.functions.convert_structures_to_population import structures_to_population
    pop = structures_to_population(structs)

    # evaluate_population requires bands; pass an empty tuple so cost = 0, but we'll replace it
    results = evaluate_population(env, pop, materials, target, ())

    # Now plot each using the existing plot_best_structure, but override the cost
    for rank, result in enumerate(results, start=1):
        # Create a new EvaluationResult with historical cost
        result_with_cost = EvaluationResult(
            structure=result.structure,
            cost=hist_costs[rank-1],
            reflectance=result.reflectance,
            transmittance=result.transmittance,
            absorptance=result.absorptance,
        )
        print(f"\nTop {rank}: cost = {hist_costs[rank-1]:.6f}")
        plot_best_structure(
            result=result_with_cost,
            env=env,
            target=target,
            material_database=materials,
            generation=None,
            save_path=None,
        )