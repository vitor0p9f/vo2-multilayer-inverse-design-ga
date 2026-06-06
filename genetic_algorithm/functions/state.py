import json
from pathlib import Path
from typing import List, Tuple
import jax.numpy as jnp
from specifications.objects.structure import Structure
from genetic_algorithm.types.evaluation_result import EvaluationResult


def update_best5(
    current_best5: List[Tuple[Structure, float]],
    results: List[EvaluationResult],
) -> List[Tuple[Structure, float]]:
    """Merge the current top‑5 archive with a new generation's results."""
    all_candidates = current_best5.copy()
    for r in results:
        all_candidates.append((r.structure, float(r.cost)))
    all_candidates.sort(key=lambda x: x[1])
    return all_candidates[:5]


def save_best5(directory: Path, best5: List[Tuple[Structure, float]]) -> None:
    """Save the all‑time best‑5 archive to best5.json."""
    out = []
    for s, cost in best5:
        out.append({
            "materials": s.materials.tolist(),
            "thicknesses_m": s.thicknesses_m.tolist(),
            "active_mask": s.active_mask.tolist(),
            "cost": cost,
        })
    with open(directory / "best5.json", "w") as f:
        json.dump(out, f, indent=2)


def save_elites(
    directory: Path,
    results: List[EvaluationResult],
    elite_frac: float,
) -> None:
    """Save the final generation's elites (elite_frac * pop_size) to elite_final.json."""
    pop_size = len(results)
    elite_count = max(1, int(elite_frac * pop_size))
    sorted_results = sorted(results, key=lambda r: r.cost)[:elite_count]
    out = []
    for r in sorted_results:
        out.append({
            "materials": r.structure.materials.tolist(),
            "thicknesses_m": r.structure.thicknesses_m.tolist(),
            "active_mask": r.structure.active_mask.tolist(),
            "cost": float(r.cost),
        })
    with open(directory / "elite_final.json", "w") as f:
        json.dump(out, f, indent=2)


def save_cost_history(directory, history):
    with open(directory / "cost_history.json", "w") as f:
        json.dump([float(x) for x in history], f, indent=2)


def load_cost_history(directory: Path) -> List[float]:
    """Load the cost history from file, or return empty list if not found."""
    path = directory / "cost_history.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []