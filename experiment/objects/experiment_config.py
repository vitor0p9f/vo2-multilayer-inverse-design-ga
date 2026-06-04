import hashlib
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Type, Union
import jax.numpy as jnp
import numpy as np

from specifications.types.optics import (
    SpectralBand, Reflectance, Transmittance, Absorptance
)
from specifications.types.structure import Thicknesses
from template.objects.template import Template
from specifications.objects.material import Material
from specifications.types.environment import Temperatures, Wavelengths, Angles

# ----------------------------------------------------------------------
# Default experiments root directory (can be changed before use)
# ----------------------------------------------------------------------
EXPERIMENTS_DIR = Path("./experiments")

# ----------------------------------------------------------------------
# Type aliases & mappings
# ----------------------------------------------------------------------
TargetClassType = Union[Type[Reflectance], Type[Transmittance], Type[Absorptance]]
TARGET_CLASS_MAP = {
    "Reflectance": Reflectance,
    "Transmittance": Transmittance,
    "Absorptance": Absorptance,
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete specification of a GA‑driven multilayer design experiment."""

    seed: int

    # Materials & structure
    materials: List[Material]
    thickness_options: Thicknesses
    template: Template

    # Environment
    temperatures: Temperatures
    wavelengths: Wavelengths
    angles: Angles

    # Target (super‑Gaussian)
    target_class: TargetClassType
    target_center: float
    target_width: float
    target_order: float
    target_amplitude: float
    target_baseline: float

    # Spectral bands for loss
    bands: List[SpectralBand]

    # GA parameters
    population_size: int
    max_generations: int
    cost_threshold: float
    elite_fraction: float
    tournament_size: int
    mutation_rate: float
    crossover_rate: float
    crossover_points: int

    def __post_init__(self):
        """Compute a stable hash from **all** configuration fields."""
        d = self.to_dict()                     # includes seed
        json_str = json.dumps(d, sort_keys=True, default=str)
        hash_val = hashlib.sha256(json_str.encode("utf-8")).hexdigest()[:12]
        object.__setattr__(self, "_hash", hash_val)

    @property
    def get_path(self) -> Path:
        """Return the experiment folder: EXPERIMENTS_DIR / hash."""
        return EXPERIMENTS_DIR / self._hash

    @property
    def hash(self) -> str:
        """Return the computed configuration hash."""
        return self._hash

    # ------------------------------------------------------------------
    # Serialization (unchanged)
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return _to_json_compatible({
            f.name: getattr(self, f.name)
            for f in self.__dataclass_fields__.values()
        })

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        target_class_str = data["target_class"]
        target_class = TARGET_CLASS_MAP[target_class_str]

        return cls(
            seed=data["seed"],
            materials=[Material.from_dict(m) for m in data["materials"]],
            thickness_options=jnp.array(data["thickness_options"]),
            template=Template.from_dict(data["template"]),
            temperatures=jnp.array(data["temperatures"]),
            wavelengths=jnp.array(data["wavelengths"]),
            angles=jnp.array(data["angles"]),
            target_class=target_class,
            target_center=data["target_center"],
            target_width=data["target_width"],
            target_order=data["target_order"],
            target_amplitude=data.get("target_amplitude", 1.0),
            target_baseline=data.get("target_baseline", 0.0),
            bands=[SpectralBand.from_dict(b) for b in data["bands"]],
            population_size=data["population_size"],
            max_generations=data["max_generations"],
            cost_threshold=data["cost_threshold"],
            elite_fraction=data["elite_fraction"],
            tournament_size=data["tournament_size"],
            mutation_rate=data["mutation_rate"],
            crossover_rate=data.get("crossover_rate", 0.8),
            crossover_points=data.get("crossover_points", 3),
        )


# ----------------------------------------------------------------------
# JSON conversion helper
# ----------------------------------------------------------------------
def _to_json_compatible(obj):
    if isinstance(obj, (jnp.ndarray, np.ndarray)):
        return obj.tolist()
    if isinstance(obj, (int, float, bool, str, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_json_compatible(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_json_compatible(v) for k, v in obj.items()}
    if isinstance(obj, type) and obj in (Reflectance, Transmittance, Absorptance):
        return obj.__name__
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dataclass_fields__"):
        return {f.name: _to_json_compatible(getattr(obj, f.name)) for f in obj.__dataclass_fields__.values()}
    raise TypeError(f"Unsupported type: {type(obj)}")