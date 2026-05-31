import jax.numpy as jnp
from jax import jit
import jax.lax as lax

from ..objects.material import Material   # adjust import as needed

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
    Predicts the extinction coefficient k at the target temperature
    using the thermal expansion approximation.
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
    Calculates the change in refractive index (delta n) from delta_k
    using the Kramers‑Kronig relation.

    This implementation is fully JIT‑compatible: it avoids boolean masks
    with non‑concrete shapes and uses lax.fori_loop for the integration.
    """
    # Angular frequency
    omega = omega_from_lambda(wavelength_m)

    # Sort by increasing frequency
    idx = jnp.argsort(omega)
    omega_sorted = omega[idx]
    delta_k_sorted = delta_k[idx]

    n_pts = omega_sorted.shape[0]

    # Pre‑compute numerator ω * Δk(ω)
    numerator = omega_sorted * delta_k_sorted

    def compute_delta_n(i: int) -> jnp.ndarray:
        """
        Compute delta_n at index i using the principal‑value integral.
        """
        omega_i = omega_sorted[i]

        # Integral for indices < i
        def body_left(k, acc):
            # trapezoidal contribution between k and k+1
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

    # Compute delta_n for all indices (vectorised over i)
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
    Predicts n and k at the target temperature using the above functions.
    Input arrays are on the same wavelength grid (in meters).
    Returns (n_predicted, k_predicted).
    """
    k_pred, delta_k = predict_temperature_based_k(
        reference_k, cte, target_temperature, reference_temperature
    )
    delta_n = kramers_kronig_delta_n(reference_wavelengths_m, delta_k)
    n_pred = reference_n + delta_n
    return n_pred, k_pred

def thermal_prediction(material: Material, T: float):
    # 1. Find index of the closest reference temperature
    ref_temps = jnp.array(material.temperatures)
    idx = jnp.argmin(jnp.abs(ref_temps - T)).item()
    T_ref = float(ref_temps[idx])

    # 2. Get the wavelength grid FOR THAT SPECIFIC REFERENCE TEMPERATURE
    wl_m = material.wavelengths_raw[idx]   # always the correct array

    # 3. Get reference n and k on the same grid
    n_ref = material.n_values_raw[idx]
    k_ref = material.k_values_raw[idx]

    # 4. Predict using the thermal optics functions
    n_pred, k_pred = predict_nk_from_reference(
        reference_wavelengths_m=wl_m,
        reference_n=n_ref,
        reference_k=k_ref,
        cte=material.cte,
        target_temperature=T,
        reference_temperature=T_ref,
    )
    
    return wl_m, n_pred, k_pred