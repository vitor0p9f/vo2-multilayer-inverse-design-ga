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
    delta_k = reference_k * (-3.0 * linear_cte) * delta_T
    k_pred = reference_k + delta_k
    return k_pred, delta_k


@jit
def kramers_kronig_delta_n(
    wavelength_m: jnp.ndarray,
    delta_k: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the change in refractive index (delta n) from delta_k
    via the Kramers‑Kronig relation.

    This implementation is fully JIT‑compatible: it avoids boolean masks
    with non‑concrete shapes and uses lax.fori_loop for the principal‑value
    integration.
    """
    omega = omega_from_lambda(wavelength_m)

    # Sort by increasing frequency
    idx = jnp.argsort(omega)
    omega_sorted = omega[idx]
    delta_k_sorted = delta_k[idx]

    n_pts = omega_sorted.shape[0]
    numerator = omega_sorted * delta_k_sorted  # ω Δk(ω)

    def compute_delta_n(i: int) -> jnp.ndarray:
        """Compute delta_n at index i using the principal‑value integral."""
        omega_i = omega_sorted[i]

        # Integral for indices < i
        def body_left(k, acc):
            x0, x1 = omega_sorted[k], omega_sorted[k+1]
            y0 = numerator[k] / (x0*x0 - omega_i*omega_i)
            y1 = numerator[k+1] / (x1*x1 - omega_i*omega_i)
            area = 0.5 * (y0 + y1) * (x1 - x0)
            return acc + area

        integral_left = lax.fori_loop(0, i, body_left, 0.0) if i > 0 else 0.0

        # Integral for indices > i
        def body_right(k, acc):
            x0, x1 = omega_sorted[k], omega_sorted[k+1]
            y0 = numerator[k] / (x0*x0 - omega_i*omega_i)
            y1 = numerator[k+1] / (x1*x1 - omega_i*omega_i)
            area = 0.5 * (y0 + y1) * (x1 - x0)
            return acc + area

        integral_right = lax.fori_loop(i+1, n_pts-1, body_right, 0.0) if i < n_pts-1 else 0.0

        return (2.0 / jnp.pi) * (integral_left + integral_right)

    # Compute delta_n for all spectral points
    delta_n_sorted = jnp.array([compute_delta_n(i) for i in range(n_pts)])

    # Restore original order
    inverse_idx = jnp.argsort(idx)
    return delta_n_sorted[inverse_idx]


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