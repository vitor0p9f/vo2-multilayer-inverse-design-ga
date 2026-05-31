from typing import Callable, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
from jax.tree_util import register_dataclass
from ..functions.load_nk_data_from_csv import load_nk_data_from_csv
from ..types.optics import Wavelengths, N_values, K_values

import re
import jax.numpy as jnp
import numpy as np

PredictionFunction = Callable[["Material", float], Tuple[Wavelengths, N_values, K_values]]

@register_dataclass
@dataclass(frozen=True)
class Material:
    """
    Immutable optical material with temperature‑dependent refractive index.

    Supports different wavelength grids for each temperature (ragged data).
    **Wavelengths are stored internally in meters**.

    Automatically caches data in the 'database' folder:
      - CSV files per temperature under ``database/csv/`` (wavelengths in meters)
      - A single NPZ file under ``database/numpy/``

    Attributes:
        name: Human‑readable name.
        symbol: Chemical symbol (used for file names).
        cte: Thermal expansion coefficient (1/K).
        files: Single Path or tuple of Paths to CSV files (assumed to contain wavelengths in µm).
        temperatures: Tuple of temperatures (K) – automatically populated.
        database_dir: Directory for caching.
    """
    name: str
    symbol: str
    cte: float
    files: Union[Path, Tuple[Path, ...]] = ()
    temperatures: Tuple[float, ...] = ()
    database_dir: Path = Path("database")

    def __post_init__(self):
        # Normalize single file to tuple
        if isinstance(self.files, Path):
            object.__setattr__(self, 'files', (self.files,))

        if self.files:
            # Extract temperatures from filenames
            temp_list = [self._extract_temperature(f) for f in self.files]
            temp_tuple = tuple(temp_list)

            # Try to load from cache first
            cached = self._load_cache_data(self.symbol, self.database_dir, temp_tuple)
            if cached is not None:
                wl_list, n_list, k_list = cached
                self._set_attrs(wl_list, n_list, k_list, temp_tuple)
                return

            # Cache miss: load from CSV files (wavelengths in µm) and convert to meters
            wl_list, n_list, k_list = [], [], []
            for f in self.files:
                wl_um, n, k = load_nk_data_from_csv(f)          # expects µm
                wl_m = jnp.asarray(wl_um, dtype=jnp.float32) * 1e-6   # convert to meters
                wl_list.append(wl_m)
                n_list.append(jnp.asarray(n, dtype=jnp.float32))
                k_list.append(jnp.asarray(k, dtype=jnp.float32))

            self._set_attrs(tuple(wl_list), tuple(n_list), tuple(k_list), temp_tuple)
            self._save_to_database()
        else:
            # No files provided – temperatures must already be set (used by from_prediction)
            if not self.temperatures:
                raise ValueError(
                    "Either 'files' or 'temperatures' must be provided. "
                    "Use .from_prediction() when no files are available."
                )

    def _set_attrs(self, wl_tuple, n_tuple, k_tuple, temps_tuple):
        """Set internal attributes and determine if wavelength grid is common."""
        object.__setattr__(self, '_wavelengths', wl_tuple)   # each element in meters
        object.__setattr__(self, '_n_values', n_tuple)
        object.__setattr__(self, '_k_values', k_tuple)
        object.__setattr__(self, 'temperatures', temps_tuple)
        self._set_common_grid_flag()

    def _set_common_grid_flag(self):
        """Check if all temperatures share exactly the same wavelength grid."""
        wl_list = self._wavelengths
        if len(wl_list) <= 1:
            is_common = True
        else:
            first = wl_list[0]
            is_common = all(
                wl.shape == first.shape and jnp.allclose(wl, first)
                for wl in wl_list[1:]
            )
        object.__setattr__(self, '_is_common_grid', is_common)

    @staticmethod
    def _extract_temperature(filepath: Path) -> float:
        """Parse temperature (Kelvin) from filename: <anything>_<number>K.csv"""
        stem = filepath.stem
        match = re.search(r'_(\d+(?:\.\d+)?)K$', stem)
        if not match:
            raise ValueError(f"Could not parse temperature from '{filepath.name}'. "
                             f"Expected format: '<symbol>_<temperature>K.csv'.")
        return float(match.group(1))

    @staticmethod
    def _load_cache_data(symbol, database_dir, temps_tuple):
        """Load cached NPZ data if it exists and temperatures match exactly."""
        npz_path = database_dir / "numpy" / f"{symbol}.npz"
        if not npz_path.exists():
            return None
        data = np.load(npz_path)
        stored_temps = tuple(data['temperatures'])
        if stored_temps != temps_tuple:
            return None
        num = int(data['num_temps'])
        wl_list = [jnp.array(data[f'wl_{i}'], dtype=jnp.float32) for i in range(num)]
        n_list  = [jnp.array(data[f'n_{i}'], dtype=jnp.float32) for i in range(num)]
        k_list  = [jnp.array(data[f'k_{i}'], dtype=jnp.float32) for i in range(num)]
        return tuple(wl_list), tuple(n_list), tuple(k_list)

    def _save_to_database(self):
        """Save material data to CSV (one per temperature) and a single NPZ file."""
        db = self.database_dir
        db.mkdir(parents=True, exist_ok=True)
        csv_dir = db / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)

        # Save one CSV per temperature (wavelengths in meters, scientific notation)
        for i, T in enumerate(self.temperatures):
            wl = np.asarray(self._wavelengths[i])   # meters
            n = np.asarray(self._n_values[i])
            k = np.asarray(self._k_values[i])
            temp_str = f"{int(T)}K" if T == int(T) else f"{T}K"
            filename = f"{self.symbol}_{temp_str}.csv"
            data = np.column_stack((wl, n, k))
            np.savetxt(
                csv_dir / filename,
                data,
                delimiter=',',
                header='wavelength_m,n,k',
                comments='',
                fmt='%.6e'          # scientific notation
            )

        # Save a single NPZ file with all data (ragged storage)
        npz_dir = db / "numpy"
        npz_dir.mkdir(parents=True, exist_ok=True)
        npz_dict = {
            'temperatures': np.array(self.temperatures),
            'num_temps': len(self.temperatures)
        }
        for i in range(len(self.temperatures)):
            npz_dict[f'wl_{i}'] = np.asarray(self._wavelengths[i])
            npz_dict[f'n_{i}'] = np.asarray(self._n_values[i])
            npz_dict[f'k_{i}'] = np.asarray(self._k_values[i])
        np.savez(npz_dir / f"{self.symbol}.npz", **npz_dict)

    @property
    def n_values(self) -> Union[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
        """Real refractive index. Returns stacked array if common grid, else tuple."""
        if self._is_common_grid:
            return jnp.stack(self._n_values)
        return self._n_values

    @property
    def k_values(self) -> Union[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
        """Extinction coefficient. Returns stacked array if common grid, else tuple."""
        if self._is_common_grid:
            return jnp.stack(self._k_values)
        return self._k_values

    @property
    def wavelengths(self) -> Union[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
        """Wavelengths in meters. Returns 1D array if common grid, else tuple."""
        if self._is_common_grid:
            return self._wavelengths[0]
        return self._wavelengths

    @property
    def n_values_raw(self) -> Tuple[jnp.ndarray, ...]:
        """Always return a tuple of n arrays (one per temperature)."""
        return self._n_values

    @property
    def k_values_raw(self) -> Tuple[jnp.ndarray, ...]:
        """Always return a tuple of k arrays (one per temperature)."""
        return self._k_values

    @property
    def wavelengths_raw(self) -> Tuple[jnp.ndarray, ...]:
        """Always return a tuple of wavelength arrays (one per temperature)."""
        return self._wavelengths

    def get_at_temperature(self, idx: int):
        """Return (wavelengths, n, k) for the temperature at index `idx`."""
        return self._wavelengths[idx], self._n_values[idx], self._k_values[idx]

    def from_prediction(
        self,
        target_temperatures: jnp.ndarray,
        prediction_function: PredictionFunction,
    ) -> "Material":
        """
        Extend the current Material by predicting optical properties at new temperatures.

        The original temperatures are kept, and new temperatures that are not already
        present are added. The prediction function is called only for the new ones.

        Args:
            target_temperatures: 1D array of temperatures (K) to predict (may include
                                 temperatures that already exist – duplicates are ignored).
            prediction_function: Function with signature:
                prediction_function(material: Material, T: float) -> (wavelengths_m, n, k)

        Returns:
            A new Material instance containing both original and predicted temperatures.
        """
        # Convert to Python floats and filter out already existing temperatures
        existing_set = set(self.temperatures)
        new_temps = []
        for T in target_temperatures:
            Tf = float(T)
            if Tf not in existing_set:
                new_temps.append(Tf)

        if not new_temps:
            # No new temperatures – return self (immutable, so it's fine)
            return self

        # Predict for each new temperature
        new_wl_list, new_n_list, new_k_list = [], [], []
        for T in new_temps:
            wl_m, n_arr, k_arr = prediction_function(self, T)
            new_wl_list.append(jnp.asarray(wl_m, dtype=jnp.float32))
            new_n_list.append(jnp.asarray(n_arr, dtype=jnp.float32))
            new_k_list.append(jnp.asarray(k_arr, dtype=jnp.float32))

        # Combine old and new data
        combined_temps = tuple(list(self.temperatures) + new_temps)
        combined_wl = self._wavelengths + tuple(new_wl_list)
        combined_n  = self._n_values + tuple(new_n_list)
        combined_k  = self._k_values + tuple(new_k_list)

        # Create a new Material with the combined data
        new_material = Material(
            name=self.name,
            symbol=self.symbol,
            cte=self.cte,
            files=(),                           # no CSV files – data passed directly
            temperatures=combined_temps,
            database_dir=self.database_dir,
        )
        new_material._set_attrs(combined_wl, combined_n, combined_k, combined_temps)
        new_material._save_to_database()
        return new_material
    
    def get_at_temperature(self, temperature: float, exact: bool = False, tol: float = 1e-6) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Return (wavelengths, n, k) for the given temperature.

        Args:
            temperature: Target temperature in Kelvin.
            exact: If True, raise KeyError if T not exactly in temperatures.
                If False (default), return the closest temperature.
            tol: Tolerance for exact match (ignored if exact=False).

        Returns:
            Tuple (wavelengths, n, k) as JAX arrays.
        """
        temps = np.array(self.temperatures)   # use numpy for simplicity
        if exact:
            mask = np.isclose(temps, temperature, atol=tol)
            if not np.any(mask):
                raise KeyError(f"Temperature {temperature}K not found in {self.temperatures}")
            idx = np.argmax(mask).item()
        else:
            idx = np.argmin(np.abs(temps - temperature)).item()
        
        return self._get_at_temperature_index(idx)

    def _get_at_temperature_index(self, idx: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return (wavelengths, n, k) for the temperature at index `idx`."""
        return self._wavelengths[idx], self._n_values[idx], self._k_values[idx]