from tmm.objects.optics import Absorptance, Reflectance, Transmittance

import jax
import jax.numpy as jnp


def tmm_single(
    n: jax.Array,          # (L,)  real part of refractive index
    k: jax.Array,          # (L,)  extinction coefficient
    d: jax.Array,          # (L,)  layer thicknesses in meters
    wavelength: float,     # scalar wavelength in meters
    theta_inc: float,      # scalar incidence angle in radians
) -> tuple[Reflectance, Transmittance, Absorptance]:
    """
    Transfer Matrix Method (TMM) for a single wavelength and single angle.

    All calculations are performed in **float64 / complex128** to
    minimise numerical errors. Input arrays and scalars are promoted
    to the highest precision available in JAX.

    Returns
    -------
    (Reflectance, Transmittance, Absorptance) :
        Each object contains s‑polarised, p‑polarised and unpolarised
        values as scalar (0‑dim) float64 JAX arrays.
    """
    # ================================================================
    # 1. Promote everything to double precision
    # ================================================================
    n = jnp.asarray(n, dtype=jnp.float64)
    k = jnp.asarray(k, dtype=jnp.float64)
    d = jnp.asarray(d, dtype=jnp.float64)
    wavelength = jnp.float64(wavelength)
    theta_inc = jnp.float64(theta_inc)

    L = n.shape[0]

    # ---- Complex refractive index and relative permittivity ----
    N = n + 1j * k                    # complex128 because n,k are float64
    E = N * N

    # ---- Propagation angles (Snell's law, complex) ----
    sin_sq = (N[0].real / N) ** 2 * jnp.sin(theta_inc) ** 2
    cos_theta = jnp.sqrt(1.0 - sin_sq)

    # ---- Normal component of the wave vector ----
    K = 2 * jnp.pi * N * cos_theta / wavelength   # complex128

    # ---- First interface matrix (layer 0 → 1) ----
    K0, K1 = K[0], K[1]
    # s‑polarisation
    M_s = jnp.array([[K0 + K1, K0 - K1],
                     [K0 - K1, K0 + K1]], dtype=jnp.complex128) / (2 * K0)

    E0, E1 = E[0], E[1]
    # p‑polarisation
    Mp = jnp.array([[K0/E0 + K1/E1, K0/E0 - K1/E1],
                    [K0/E0 - K1/E1, K0/E0 + K1/E1]], dtype=jnp.complex128) / (2 * K0 / E0)

    # ---- Propagate through interior layers (2 … L-1) ----
    for m in range(2, L):
        K_m = K[m-1]
        D_m = d[m-1]
        # Phase matrix for a homogeneous layer
        P = jnp.array([[jnp.exp(-1j * K_m * D_m), 0j],
                       [0j, jnp.exp(1j * K_m * D_m)]], dtype=jnp.complex128)

        M_s = P @ M_s
        Mp  = P @ Mp

        K_cur = K[m-1]
        K_next = K[m]
        E_cur = E[m-1]
        E_next = E[m]

        # s‑polarisation interface
        M_int_s = jnp.array([[K_cur + K_next, K_cur - K_next],
                             [K_cur - K_next, K_cur + K_next]], dtype=jnp.complex128) / (2 * K_cur)
        M_s = M_int_s @ M_s

        # p‑polarisation interface
        M_int_p = jnp.array([[K_cur/E_cur + K_next/E_next, K_cur/E_cur - K_next/E_next],
                             [K_cur/E_cur - K_next/E_next, K_cur/E_cur + K_next/E_next]], dtype=jnp.complex128) / (2 * K_cur / E_cur)
        Mp = M_int_p @ Mp

    # ---- Amplitude coefficients ----
    r_s = M_s[1, 0] / M_s[0, 0]
    t_s = 1.0 / M_s[0, 0]
    r_p = Mp[1, 0] / Mp[0, 0]
    t_p = 1.0 / Mp[0, 0]

    # ---- Power normalisation factors ----
    n_inc = n[0]
    n_sub = n[-1]
    cos_inc = jnp.abs(jnp.cos(theta_inc))

    # Cosine of exit angle – use complex sqrt to avoid NaN in TIR
    cos_sub_complex = jnp.sqrt(1.0 - (n_inc / n_sub * jnp.sin(theta_inc))**2 + 0j)
    cos_sub = cos_sub_complex.real   # zero under total internal reflection

    T_factor_s = (n_sub * cos_sub) / (n_inc * cos_inc)
    T_factor_p = (n_inc * cos_sub) / (n_sub * cos_inc)

    # ---- Reflectance, Transmittance, Absorptance ----
    R_s = jnp.abs(r_s) ** 2
    T_s = T_factor_s * jnp.abs(t_s) ** 2
    A_s = 1.0 - R_s - T_s

    R_p = jnp.abs(r_p) ** 2
    T_p = T_factor_p * jnp.abs(t_p) ** 2
    A_p = 1.0 - R_p - T_p

    # Unpolarised = average of s and p
    R_un = 0.5 * (R_s + R_p)
    T_un = 0.5 * (T_s + T_p)
    A_un = 0.5 * (A_s + A_p)

    return (
        Reflectance(p_polarized=R_p, s_polarized=R_s, non_polarized=R_un),
        Transmittance(p_polarized=T_p, s_polarized=T_s, non_polarized=T_un),
        Absorptance(p_polarized=A_p, s_polarized=A_s, non_polarized=A_un),
    )