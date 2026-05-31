"""
Edlén equation for the refractive index of air.

Implements the modified Edlén (1966) formula with humidity and pressure
corrections.  The extinction coefficient k is assumed to be zero.
"""

import jax.numpy as jnp
from jax import jit
from structure.objects.material import Material


@jit
def water_saturation_vapor_pressure(temperature_K: jnp.ndarray) -> jnp.ndarray:
    """
    Saturation vapor pressure of water (Pa) for given Kelvin temperatures.
    Vectorised – accepts an array of temperatures.
    """
    # Constants of the equation
    K1 = 1167.05214528
    K2 = -724213.167032
    K3 = -17.0738469401
    K4 = 12020.8247025
    K5 = -3232555.03223
    K6 = 14.9151086135
    K7 = -4823.26573616
    K8 = 405113.405421
    K9 = -0.238555575678
    K10 = 650.175348448

    T = temperature_K
    omega = T + (K9 / (T - K10))
    omega2 = omega ** 2

    A = omega2 + K1 * omega + K2
    B = K3 * omega2 + K4 * omega + K5
    C = K6 * omega2 + K7 * omega + K8

    X = -B + jnp.sqrt(B**2 - 4 * A * C)
    return 1e6 * ((2 * C) / X) ** 4


@jit
def edlen_n_air(
    wavelength_m: jnp.ndarray,          # 1D array of wavelengths (m)
    temperature_K: float,               # target temperature (K)
    humidity_pct: float = 50.0,         # relative humidity (%)
    pressure_Pa: float = 101325.0,      # pressure (Pa)
) -> jnp.ndarray:
    """
    Refractive index of air from the Edlén equation (modified).

    Args:
        wavelength_m: Wavelengths in meters, shape (N,).
        temperature_K: Temperature in Kelvin.
        humidity_pct: Relative humidity (0‑100).
        pressure_Pa: Atmospheric pressure in Pa.

    Returns:
        n: Refractive index, shape (N,).
    """
    # Convert to micrometres
    wl_um = wavelength_m * 1e6
    S = 1.0 / (wl_um ** 2)

    # Edlén constants
    A = 8342.54
    B = 2406147.0
    C = 15998.0
    D = 96095.43
    E = 0.601
    F = 0.00972
    G = 0.003661

    # Standard dry air refractive index (15 °C, 101325 Pa)
    ns = 1.0 + 1e-8 * (A + B / (130.0 - S) + C / (38.9 - S))

    # Saturation vapor pressure at the target temperature
    p_sat = water_saturation_vapor_pressure(temperature_K)  # scalar

    # Partial pressure of water vapour
    pv = (humidity_pct / 100.0) * p_sat

    # Pressure‑temperature correction for dry air
    t = temperature_K
    X = (1.0 + 1e-8 * (E - F * t) * pressure_Pa) / (1.0 + G * t)

    # Dry air refractive index at actual conditions
    n_dry = 1.0 + (pressure_Pa * (ns - 1.0) * X) / D

    humidity_correction = (
        1e-10
        * (292.75 / t)
        * (3.7345 - 0.0401 * S)
        * pv
    )

    return n_dry - humidity_correction


def edlen_prediction_batch(
    material: Material,
    temperatures: jnp.ndarray,   # 1D array of target temperatures (K)
    humidity_pct: float = 0.0,
    pressure_Pa: float = 101325.0,
) -> tuple[tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...]]:
    """
    Batch prediction of air refractive index using Edlén's equation.

    For each target temperature, the wavelength grid of the closest existing
    temperature in the material is used.  The extinction coefficient k is
    always zero.

    Args:
        material: Material instance (its n/k values are ignored; only
                  wavelengths and temperatures are used).
        temperatures: 1D JAX array of target temperatures (K).
        humidity_pct: Relative humidity in %.
        pressure_Pa: Atmospheric pressure in Pa.

    Returns:
        wavelengths_tuple: tuple of 1D wavelength arrays (one per target T).
        n_tuple: tuple of 1D n arrays (one per target T).
        k_tuple: tuple of 1D k arrays (all zeros).
    """
    ref_temps = jnp.array(material.temperatures)
    wl_out = []
    n_out = []
    k_out = []

    for T in temperatures:
        T_float = float(T)
        idx = jnp.argmin(jnp.abs(ref_temps - T)).item()
        wl_m = material.wavelengths_raw[idx]

        n_pred = edlen_n_air(
            wavelength_m=wl_m,
            temperature_K=T_float,
            humidity_pct=humidity_pct,
            pressure_Pa=pressure_Pa,
        )
        k_pred = jnp.zeros_like(n_pred)

        wl_out.append(wl_m)
        n_out.append(n_pred)
        k_out.append(k_pred)

    return tuple(wl_out), tuple(n_out), tuple(k_out)


# Optional single‑temperature wrapper
def edlen_prediction(material: Material, T: float) -> tuple:
    """Single‑temperature prediction (backward compatible)."""
    wl_tuple, n_tuple, k_tuple = edlen_prediction_batch(
        material, jnp.array([T])
    )
    return wl_tuple[0], n_tuple[0], k_tuple[0]