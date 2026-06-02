from tmm.objects.optics import Absorptance, Reflectance, Transmittance

import jax
import jax.numpy as jnp

def tmm_single(
    n: jax.Array,          # (L,)
    k: jax.Array,          # (L,)
    d: jax.Array,          # (L,)  (ignored for first and last layers)
    wavelength: float,     # scalar
    theta_inc: float,      # scalar
) -> tuple[Reflectance, Transmittance, Absorptance]:
    """
    Transfer Matrix Method for one wavelength and one incidence angle.
    Returns typed containers with scalar (0‑dim) fields.
    """

    L = n.shape[0]

    # --- Complex refractive index & permittivity ---
    N = n + 1j * k                   # (L,)
    E = N * N

    # --- cos(theta_m) from Snell's law ---
    sin_sq = (N[0].real / N) ** 2 * jnp.sin(theta_inc) ** 2
    cos_theta = jnp.sqrt(1.0 - sin_sq)   # complex, (L,)

    # --- Normal component of wave vector ---
    K = 2 * jnp.pi * N * cos_theta / wavelength   # (L,)

    # --- Initialise total transfer matrix with M_{2,1} ---
    K0, K1 = K[0], K[1]
    M_s = jnp.array([[K0 + K1, K0 - K1],
                     [K0 - K1, K0 + K1]]) / (2 * K0)

    E0, E1 = E[0], E[1]
    Mp = jnp.array([[K0/E0 + K1/E1, K0/E0 - K1/E1],
                    [K0/E0 - K1/E1, K0/E0 + K1/E1]]) / (2 * K0 / E0)

    # --- Loop over interior layers (2 … L-1) ---
    for m in range(2, L):
        # Propagation through layer m (index m-1 in 0‑based)
        K_m = K[m-1]
        D_m = d[m-1]
        P = jnp.array([[jnp.exp(-1j * K_m * D_m), 0j],
                       [0j, jnp.exp(1j * K_m * D_m)]])

        M_s = P @ M_s
        Mp  = P @ Mp

        # Interface matrix between m and m+1 (indices m-1 → m)
        K_cur = K[m-1]
        K_next = K[m]
        E_cur = E[m-1]
        E_next = E[m]

        # s‑pol
        M_int_s = jnp.array([[K_cur + K_next, K_cur - K_next],
                             [K_cur - K_next, K_cur + K_next]]) / (2 * K_cur)
        M_s = M_int_s @ M_s

        # p‑pol
        M_int_p = jnp.array([[K_cur/E_cur + K_next/E_next, K_cur/E_cur - K_next/E_next],
                             [K_cur/E_cur - K_next/E_next, K_cur/E_cur + K_next/E_next]]) / (2 * K_cur / E_cur)
        Mp = M_int_p @ Mp

    # --- Reflection & transmission amplitude coefficients ---
    r_s = M_s[1, 0] / M_s[0, 0]
    t_s = 1.0 / M_s[0, 0]
    r_p = Mp[1, 0] / Mp[0, 0]
    t_p = 1.0 / Mp[0, 0]

    # --- Incidence and exit medium parameters (assumed lossless) ---
    n_inc = n[0]                 # real
    n_sub = n[-1]                # real
    cos_inc = jnp.abs(jnp.cos(theta_inc))                # real
    cos_sub = jnp.sqrt(1 - (n_inc / n_sub * jnp.sin(theta_inc)) ** 2).real

    # --- Correct normalisation factors ---
    T_factor_s = (n_sub * cos_sub) / (n_inc * cos_inc)   # for s‑polarisation
    T_factor_p = (n_inc * cos_sub) / (n_sub * cos_inc)   # for p‑polarisation

    # --- Reflectance, Transmittance, Absorptance ---
    R_s = jnp.abs(r_s) ** 2
    T_s = T_factor_s * jnp.abs(t_s) ** 2
    A_s = 1 - R_s - T_s

    R_p = jnp.abs(r_p) ** 2
    T_p = T_factor_p * jnp.abs(t_p) ** 2
    A_p = 1 - R_p - T_p

    R_un = 0.5 * (R_s + R_p)
    T_un = 0.5 * (T_s + T_p)
    A_un = 0.5 * (A_s + A_p)

    return (
        Reflectance(p_polarized=R_p, s_polarized=R_s, non_polarized=R_un),
        Transmittance(p_polarized=T_p, s_polarized=T_s, non_polarized=T_un),
        Absorptance(p_polarized=A_p, s_polarized=A_s, non_polarized=A_un),
    )