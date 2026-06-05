from typing import List, Tuple, Union
import jax
import jax.numpy as jnp
from jaxtyping import Float32, Int8, Array

from specifications.types.environment import Environment, Temperatures, Wavelengths
from specifications.types.optics import SpectralBand, Reflectance, Transmittance, Absorptance
from specifications.objects.structure import Structure
from ..objects.population import Population
from ..types.evaluation_result import EvaluationResult
from specifications.objects.material import Material
from tmm.functions.batch_tmm import tmm_batch
from specifications.types.structure import Thicknesses, Mask


def evaluate_population(
    env: Environment,
    pop: Population,
    material_database: List[Material],
    target: Union[Reflectance, Transmittance, Absorptance],
    bands: Tuple[SpectralBand, ...],
) -> List[EvaluationResult]:
    """
    Evaluate every structure in the population under the given environment.

    For each structure, the optical response (reflectance, transmittance,
    absorptance) is computed over all temperatures and angles, and a scalar
    cost is calculated using a weighted band‑limited MSE against the target.

    The active_mask of each structure is respected **only for free layers**:
    free layers with active_mask=False are given zero thickness, effectively
    removing them from the optical stack. Fixed layers (incidence, substrate,
    and any layers not marked as free) always retain their thickness.

    Args:
        env:              Environment (temperatures, angles, wavelengths).
        pop:              Population of structures (batched arrays).
        material_database: Available materials.
        target:           Target optical property.
        bands:            Spectral bands with weights for the loss.

    Returns:
        List of EvaluationResult, one per structure in the population.
    """
    # ==================================================================
    # 1.  Pre‑compute n,k for all materials on the common wavelength grid
    #     at every temperature → (T, num_materials, W)
    # ==================================================================
    num_materials = len(material_database)
    W = env.wavelengths.shape[0]
    T = env.temperatures.shape[0]

    n_all = jnp.zeros((T, num_materials, W), dtype=jnp.float32)
    k_all = jnp.zeros((T, num_materials, W), dtype=jnp.float32)

    for t_idx, temp in enumerate(env.temperatures):
        for m_idx, mat in enumerate(material_database):
            wl_mat, n_vals, k_vals = mat.get_data_by_temperature(temp)
            n_all = n_all.at[t_idx, m_idx].set(
                jnp.interp(env.wavelengths, wl_mat, n_vals)
            )
            k_all = k_all.at[t_idx, m_idx].set(
                jnp.interp(env.wavelengths, wl_mat, k_vals)
            )

    # Free masks are identical for all individuals – take from the first one.
    free_mat_mask = pop.free_material_mask[0]
    free_thick_mask = pop.free_thickness_mask[0]
    free_mask = free_mat_mask | free_thick_mask   # layers that may be turned off

    # ==================================================================
    # 2.  Nested loss function (mask‑based, no boolean indexing issues)
    # ==================================================================
    def _compute_loss(
        computed: Float32[Array, "T A W"],
        target_nonpol: Float32[Array, "1 W"],
    ) -> Float32[Array, ""]:
        """
        Weighted band‑limited MSE, combined as mean + std + max over T.
        Uses float masks and explicit sums to avoid boolean indexing.
        """
        # target_nonpol: (1, W) → expand to (1, 1, W) for broadcasting
        target_expanded = target_nonpol[jnp.newaxis, jnp.newaxis, :]  # (1, 1, W)

        def _per_temp(spectrum_T):   # spectrum_T: (A, W)
            loss = jnp.float32(0.0)
            for band in bands:
                # Float mask: 1.0 inside band, 0.0 outside
                mask = band.contains(env.wavelengths).astype(jnp.float32)  # (W,)
                # Squared error: (A, W) vs (1, 1, W) → broadcast
                sq_err = (spectrum_T - target_expanded[0, :, :]) ** 2
                # Band MSE = sum(sq_err * mask) / sum(mask)
                band_mse = jnp.sum(sq_err * mask, axis=-1) / jnp.sum(mask)  # (A,)
                # Average over angles, then add weighted contribution
                loss = loss + band.weight * jnp.mean(band_mse)
            return loss

        loss_per_temp = jax.vmap(_per_temp)(computed)   # (T,)
        return jnp.mean(loss_per_temp) + jnp.std(loss_per_temp) + jnp.max(loss_per_temp)

    # ==================================================================
    # 3.  Determine which property to match (robust against class re‑definition)
    # ==================================================================
    if isinstance(target, Reflectance) or type(target).__name__ == "Reflectance":
        prop_attr = 'reflectance'
    elif isinstance(target, Transmittance) or type(target).__name__ == "Transmittance":
        prop_attr = 'transmittance'
    else:
        prop_attr = 'absorptance'

    # ==================================================================
    # 4.  Per‑structure evaluation (vmapped over the population)
    # ==================================================================
    def _eval_one_structure(
        mat_indices: Int8[Array, "L"],
        thicknesses: Thicknesses,
        active_mask: Mask,
    ) -> Tuple[
        Float32[Array, ""],
        Float32[Array, "T A W"],
        Float32[Array, "T A W"],
        Float32[Array, "T A W"],
    ]:
        # Zero thickness only for free layers that are inactive.
        # Fixed layers always keep their thickness, regardless of active_mask.
        thicknesses = jnp.where(active_mask | ~free_mask, thicknesses, 0.0)

        # Select n,k for this structure's materials.
        # n_all, k_all are (T, num_materials, W), mat_indices (L,)
        # We need (T, W, L) for tmm_batch.
        n_T = n_all[:, mat_indices, :]   # (T, L, W)
        k_T = k_all[:, mat_indices, :]
        # Transpose to (T, W, L)
        n_T = jnp.transpose(n_T, (0, 2, 1))
        k_T = jnp.transpose(k_T, (0, 2, 1))

        # TMM per temperature → (T, A, W) non‑polarised
        def _tmm_one_temp(n_wl, k_wl):   # (W, L)
            R, T, A = tmm_batch(n_wl, k_wl, thicknesses, env.wavelengths, env.angles)
            return R.non_polarized, T.non_polarized, A.non_polarized

        R_all, T_all, A_all = jax.vmap(_tmm_one_temp, in_axes=(0, 0))(n_T, k_T)

        # Choose property for loss
        if prop_attr == 'reflectance':
            P = R_all
        elif prop_attr == 'transmittance':
            P = T_all
        else:
            P = A_all

        cost = _compute_loss(P, target.non_polarized)
        return cost, R_all, T_all, A_all

    # Vectorise over population – now maps over three arrays
    batch_fn = jax.vmap(_eval_one_structure, in_axes=(0, 0, 0))
    costs, R_batch, T_batch, A_batch = batch_fn(
        pop.materials, pop.thicknesses_m, pop.active_mask
    )

    # ==================================================================
    # 5.  Build EvaluationResult objects
    # ==================================================================
    results = []
    for i in range(pop.size):
        individual_struct = Structure(
            materials=pop.materials[i],
            thicknesses_m=pop.thicknesses_m[i],
            active_mask=pop.active_mask[i],
            free_thickness_mask=pop.free_thickness_mask[i],
            free_material_mask=pop.free_material_mask[i],
        )
        results.append(EvaluationResult(
            structure=individual_struct,
            cost=costs[i],
            reflectance=R_batch[i],
            transmittance=T_batch[i],
            absorptance=A_batch[i],
        ))
    return results