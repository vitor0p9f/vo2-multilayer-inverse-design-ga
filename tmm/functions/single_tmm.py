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

    Computes the reflectance, transmittance and absorptance of a plane‑
    parallel multilayer stack. The incidence medium (layer 0) and the
    substrate (layer L‑1) are assumed lossless and semi‑infinite.

    The algorithm follows the standard TMM formulation:
      - Build interface matrices (Fresnel coefficients) for s‑ and
        p‑polarisation at every boundary.
      - Multiply interface and propagation matrices from the incidence
        side to the substrate.
      - Extract the amplitude reflection/transmission coefficients and
        convert to power ratios.

    Parameters
    ----------
    n : (L,) jax.Array
        Real part of the complex refractive index for each layer.
    k : (L,) jax.Array
        Extinction coefficient (imaginary part) for each layer.
    d : (L,) jax.Array
        Thickness of each layer [m].  The values for the first and last
        layers are ignored because those media are semi‑infinite.
    wavelength : float
        Free‑space wavelength [m].
    theta_inc : float
        Angle of incidence in the incidence medium [rad].

    Returns
    -------
    (Reflectance, Transmittance, Absorptance) :
        Each object contains the s‑polarised, p‑polarised and unpolarised
        (average) values as scalar (0‑dim) JAX arrays.
    """

    L = n.shape[0]

    # ---- 1. Complex refractive index and relative permittivity ----
    N = n + 1j * k                    # (L,)
    E = N * N                         # relative permittivity (ε)

    # ---- 2. Propagation angles (Snell's law, complex) ----
    sin_sq = (N[0].real / N) ** 2 * jnp.sin(theta_inc) ** 2
    cos_theta = jnp.sqrt(1.0 - sin_sq)    # complex cosine for every layer

    # ---- 3. Normal component of the wave vector ----
    # K = 2π N cosθ / λ   (complex)
    K = 2 * jnp.pi * N * cos_theta / wavelength   # (L,)

    # ---- 4. Initialise total transfer matrix with the first interface ----
    # Interface between layer 0 (incidence) and layer 1
    K0, K1 = K[0], K[1]
    # s‑polarisation
    M_s = jnp.array([[K0 + K1, K0 - K1],
                     [K0 - K1, K0 + K1]]) / (2 * K0)

    E0, E1 = E[0], E[1]
    # p‑polarisation
    Mp = jnp.array([[K0/E0 + K1/E1, K0/E0 - K1/E1],
                    [K0/E0 - K1/E1, K0/E0 + K1/E1]]) / (2 * K0 / E0)

    # ---- 5. Propagate through interior layers (2 … L-1) ----
    for m in range(2, L):
        # Propagation through layer m (index m‑1 in 0‑based)
        K_m = K[m-1]
        D_m = d[m-1]
        # Phase matrix for a homogeneous layer
        P = jnp.array([[jnp.exp(-1j * K_m * D_m), 0j],
                       [0j, jnp.exp(1j * K_m * D_m)]])

        # Apply propagation
        M_s = P @ M_s
        Mp  = P @ Mp

        # Interface matrix between layers m‑1 and m (0‑based indices)
        K_cur = K[m-1]
        K_next = K[m]
        E_cur = E[m-1]
        E_next = E[m]

        # s‑polarisation interface
        M_int_s = jnp.array([[K_cur + K_next, K_cur - K_next],
                             [K_cur - K_next, K_cur + K_next]]) / (2 * K_cur)
        M_s = M_int_s @ M_s

        # p‑polarisation interface
        M_int_p = jnp.array([[K_cur/E_cur + K_next/E_next, K_cur/E_cur - K_next/E_next],
                             [K_cur/E_cur - K_next/E_next, K_cur/E_cur + K_next/E_next]]) / (2 * K_cur / E_cur)
        Mp = M_int_p @ Mp

    # ---- 6. Amplitude reflection/transmission coefficients ----
    # For s‑pol:  r = M₂₁/M₁₁,  t = 1/M₁₁
    r_s = M_s[1, 0] / M_s[0, 0]
    t_s = 1.0 / M_s[0, 0]
    # For p‑pol
    r_p = Mp[1, 0] / Mp[0, 0]
    t_p = 1.0 / Mp[0, 0]

    # ---- 7. Power normalisation factors ----
    # Incidence and substrate are assumed lossless (use real part of indices)
    n_inc = n[0]
    n_sub = n[-1]
    # Cosine of incidence angle (real)
    cos_inc = jnp.abs(jnp.cos(theta_inc))
    # Cosine of exit angle (real part from Snell's law)
    cos_sub = jnp.sqrt(1.0 - (n_inc / n_sub * jnp.sin(theta_inc)) ** 2).real

    # Correct factors (different for s and p due to Fresnel conventions)
    T_factor_s = (n_sub * cos_sub) / (n_inc * cos_inc)
    T_factor_p = (n_inc * cos_sub) / (n_sub * cos_inc)

    # ---- 8. Reflectance, Transmittance, Absorptance ----
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