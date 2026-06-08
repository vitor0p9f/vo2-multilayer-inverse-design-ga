from pathlib import Path
from typing import Optional, Tuple, List
import jax
import jax.numpy as jnp
from genetic_algorithm.objects.population import Population
from specifications.objects.structure import Structure
import json


def load_checkpoint(
    experiment_dir: Path,
) -> Optional[Tuple[int, Structure, float, Population, jax.Array, List[Tuple[Structure, float]]]]:
    """
    Load a previously saved checkpoint, including the best‑5 archive.

    Returns:
        (generation, best_structure, best_cost, population, key, best5)
        or None if no checkpoint exists.
    """
    ckpt_dir = experiment_dir / "checkpoint"
    state_file = ckpt_dir / "checkpoint.json"
    pop_file = ckpt_dir / "population.npz"

    if not state_file.exists() or not pop_file.exists():
        return None

    with open(state_file) as f:
        state = json.load(f)

    generation = state["generation"]
    best_cost = state["best_cost"]
    key = jnp.array(state["key"], dtype=jnp.uint32)

    # Rebuild best structure
    bs = state["best_structure"]
    active_mask_arr = jnp.array(bs["active_mask"], dtype=bool)
    best_structure = Structure(
        materials=jnp.array(bs["materials"], dtype=jnp.int8),
        thicknesses_m=jnp.array(bs["thicknesses_m"], dtype=jnp.float64),
        active_mask=active_mask_arr,
        free_thickness_mask=jnp.array(bs["free_thickness_mask"], dtype=bool),
        free_material_mask=jnp.array(bs["free_material_mask"], dtype=bool),
    )

    # Load population
    with jnp.load(pop_file) as data:
        population = Population(
            materials=data["materials"],
            thicknesses_m=data["thicknesses_m"],
            active_mask=data["active_mask"],
            free_thickness_mask=data["free_thickness_mask"],
            free_material_mask=data["free_material_mask"],
        )

    # Rebuild best‑5 archive
    best5 = []
    for item in state.get("best5", []):
        s = Structure(
            materials=jnp.array(item["materials"], dtype=jnp.int8),
            thicknesses_m=jnp.array(item["thicknesses_m"], dtype=jnp.float64),
            active_mask=jnp.array(item["active_mask"], dtype=bool),
            free_thickness_mask=jnp.array(item.get("free_thickness_mask", []), dtype=bool),
            free_material_mask=jnp.array(item.get("free_material_mask", []), dtype=bool),
        )
        best5.append((s, item["cost"]))

    return generation, best_structure, best_cost, population, key, best5