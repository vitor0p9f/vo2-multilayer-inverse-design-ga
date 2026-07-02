"""
Thermal optics model for solids.

Uses the coefficient of thermal expansion (CTE) to predict the change in
extinction coefficient k, then applies the Kramers‑Kronig relation to
obtain the corresponding change in refractive index n.
"""

import jax.numpy as jnp
from jax import jit
import jax.lax as lax
from specifications.objects.material import Material


@jit
def omega_from_lambda(wavelength_m: jnp.ndarray) -> jnp.ndarray:
    """Angular frequency (rad/s) from wavelength (m)."""
    c = 299792458.0  # m/s
    return 2.0 * jnp.pi * c / wavelength_m


@jit
def predict_temperature_based_k(
    reference_k: jnp.ndarray,
    linear_cte: float,
    target_temperature: float,
    reference_temperature: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Predict the extinction coefficient k at the target temperature
    using the linear thermal expansion approximation.

    Returns (k_predicted, delta_k).
    """
    delta_T = target_temperature - reference_temperature
    gamma = 3.0 * linear_cte
    
    # Exponential model
    k_pred = reference_k * jnp.exp(-gamma * delta_T)
    delta_k = k_pred - reference_k
    return k_pred, delta_k


@jit
def kramers_kronig_delta_n(
    wavelength_m: jnp.ndarray,   # (N,)
    delta_k: jnp.ndarray,        # (N,)
) -> jnp.ndarray:
    """
    Fast Kramers‑Kronig delta‑n via vectorised principal‑value integration.

    Complexity: O(N²) memory, but fully parallel and JIT‑friendly.
    For typical spectroscopic grids (N < ~2000) this compiles instantly
    and runs in milliseconds.
    """
    omega = omega_from_lambda(wavelength_m)

    # Sort by angular frequency (required for trapezoidal weights)
    idx = jnp.argsort(omega)
    omega_sorted = omega[idx]
    dk_sorted = delta_k[idx]
    N = omega_sorted.shape[0]

    # Numerator: ω * Δk(ω)
    num = omega_sorted * dk_sorted

    # Trapezoidal integration weights for non‑uniform grid
    dw = jnp.diff(omega_sorted)
    weights = jnp.zeros_like(omega_sorted)
    weights = weights.at[0].set(dw[0] / 2.0)
    weights = weights.at[-1].set(dw[-1] / 2.0)
    weights = weights.at[1:-1].set((dw[:-1] + dw[1:]) / 2.0)

    # Matrix of denominators: ω_j² – ω_i², shape (N, N)
    omega_sq = omega_sorted ** 2
    denom = omega_sq[:, None] - omega_sq[None, :]

    # Set diagonal to a non‑zero value to avoid NaN; we will zero the diagonal after division
    denom = denom.at[jnp.diag_indices(N)].set(1.0)

    # Element‑wise integrand: M[i,j] = (num[j] * w[j]) / (ω_j² – ω_i²)
    M = num[None, :] * weights[None, :] / denom

    # The diagonal corresponds to the singular point ω_i = ω_j, excluded from the integral
    M = M.at[jnp.diag_indices(N)].set(0.0)

    # Integral ≈ sum over j for each i
    delta_n_sorted = (2.0 / jnp.pi) * jnp.sum(M, axis=1)

    # Restore original wavelength order
    inv_idx = jnp.argsort(idx)
    return delta_n_sorted[inv_idx]

def predict_nk_from_reference(
    reference_wavelengths_m: jnp.ndarray,
    reference_n: jnp.ndarray,
    reference_k: jnp.ndarray,
    cte: float,
    target_temperature: float,
    reference_temperature: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Predict n and k at the target temperature using thermal expansion
    and Kramers‑Kronig.

    All inputs share the same wavelength grid (in meters).
    Returns (n_predicted, k_predicted).
    """
    k_pred, delta_k = predict_temperature_based_k(
        reference_k, cte, target_temperature, reference_temperature
    )
    delta_n = kramers_kronig_delta_n(reference_wavelengths_m, delta_k)
    n_pred = reference_n + delta_n
    return n_pred, k_pred


def thermal_prediction_batch(
    material: Material,
    temperatures: jnp.ndarray,   # 1D array of target temperatures (K)
) -> tuple[tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...]]:
    """
    Batch prediction of n and k for a solid using the thermal CTE model.

    For each target temperature, the closest existing temperature in the
    material is used as reference. The wavelength grid of that reference
    is adopted for the prediction.

    Args:
        material: Material instance with at least one temperature.
        temperatures: 1D JAX array of target temperatures (K).

    Returns:
        wavelengths_tuple: tuple of 1D wavelength arrays (one per target T).
        n_tuple: tuple of 1D n arrays (one per target T).
        k_tuple: tuple of 1D k arrays (one per target T).
    """
    ref_temps = jnp.array(material.temperatures)
    wl_out = []
    n_out = []
    k_out = []

    for T in temperatures:
        T_float = float(T)
        # Find closest reference temperature index
        idx = jnp.argmin(jnp.abs(ref_temps - T)).item()
        T_ref = float(ref_temps[idx])

        wl_m = material.wavelengths_raw[idx]
        n_ref = material.n_values_raw[idx]
        k_ref = material.k_values_raw[idx]

        n_pred, k_pred = predict_nk_from_reference(
            reference_wavelengths_m=wl_m,
            reference_n=n_ref,
            reference_k=k_ref,
            cte=material.cte,
            target_temperature=T_float,
            reference_temperature=T_ref,
        )
        wl_out.append(wl_m)
        n_out.append(n_pred)
        k_out.append(k_pred)

    return tuple(wl_out), tuple(n_out), tuple(k_out)


# Optional single‑temperature wrapper
def thermal_prediction(material: Material, T: float):
    """Single‑temperature prediction (kept for backward compatibility)."""
    wl_tuple, n_tuple, k_tuple = thermal_prediction_batch(
        material, jnp.array([T])
    )
    return wl_tuple[0], n_tuple[0], k_tuple[0]