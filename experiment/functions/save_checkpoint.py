import json
from pathlib import Path
from typing import List, Tuple
import jax
import jax.numpy as jnp
from genetic_algorithm.objects.population import Population
from specifications.objects.structure import Structure


def save_checkpoint(
    experiment_dir: Path,
    generation: int,
    best_structure: Structure,
    best_cost: float,
    population: Population,
    key: jax.Array,
    best5: List[Tuple[Structure, float]],   # new: top‑5 archive
) -> None:
    """
    Save the minimal state needed to resume the GA, including the
    all‑time best‑5 archive.
    """
    ckpt_dir = experiment_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Lightweight JSON
    state = {
        "generation": generation,
        "best_cost": float(best_cost),
        "best_structure": {
            "materials": best_structure.materials.tolist(),
            "thicknesses_m": best_structure.thicknesses_m.tolist(),
            "active_mask": best_structure.active_mask.tolist(),
        },
        "key": key.tolist(),
        "best5": [
            {
                "materials": s.materials.tolist(),
                "thicknesses_m": s.thicknesses_m.tolist(),
                "active_mask": s.active_mask.tolist(),
                "cost": float(c),
            }
            for s, c in best5
        ],
    }
    with open(ckpt_dir / "checkpoint.json", "w") as f:
        json.dump(state, f, indent=2)

    # Large population arrays → binary
    jnp.savez(
        ckpt_dir / "population.npz",
        materials=population.materials,
        thicknesses_m=population.thicknesses_m,
        active_mask=population.active_mask,
        free_thickness_mask=population.free_thickness_mask,
        free_material_mask=population.free_material_mask,
    )