from pathlib import Path
from typing import List, Union
from genetic_algorithm.types.evaluation_result import EvaluationResult
from specifications.types.optics import Reflectance, Transmittance, Absorptance
from specifications.objects.material import Material
from specifications.objects.environment import Environment
from .plot_best import plot_best_structure


def save_generation_data(
    directory: Path,
    generation: int,
    results: List[EvaluationResult],
    best_result: EvaluationResult,
    env: Environment,
    target: Union[Reflectance, Transmittance, Absorptance],
    materials: List[Material],
):
    """Save only the plot of the best structure for a given generation."""
    out_dir = directory / "best_structures"
    out_dir.mkdir(exist_ok=True)

    gen_str = f"{generation:04d}"
    try:
        plot_best_structure(
            result=best_result,
            env=env,
            target=target,
            material_database=materials,
            save_path=out_dir / f"best_spectrum_{gen_str}.png",
            generation=generation,
        )
    except Exception as e:
        print(f"  [Warning] Could not plot generation {generation}: {e}")