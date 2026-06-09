from typing import Tuple, List
import jax
import jax.numpy as jnp
from jax import lax

from ..objects.population import Population
from specifications.types.structure import Materials, Thicknesses, Mask
from specifications.objects.material import Material


def mutate_population(
    pop: Population,
    key: jax.Array,
    material_database: List[Material],
    thickness_options: Thicknesses,
    mutation_rate: float = 0.1,
    allow_air_gap: bool = False,
) -> Tuple[jax.Array, Population]:
    """
    Apply one randomly chosen mutation per individual.

    With probability `mutation_rate`, an individual is mutated. If mutated,
    a mutation type is chosen uniformly among:
      0 – Scramble free layers (only where material AND thickness are free)
      1 – Grow / shrink (toggle active status of fully free layers independently)
      2 – Mutate materials of free‑material layers (per‑gene with prob. `mutation_rate`)
      3 – Mutate thicknesses of free‑thickness layers (per‑gene with prob. `mutation_rate`)

    The grow/shrink operator toggles the active status of fully free layers
    independently with probability `mutation_rate`, while always keeping
    at least one fully free layer active.

    Args:
        pop:                 Population to mutate.
        key:                 JAX PRNG key.
        material_database:   List of Material objects; allowed material indices
                             are 0 … len(material_database)-1.
        thickness_options:   1D array of allowed thicknesses.
        mutation_rate:       Overall mutation probability; also used as
                             per‑gene rate for material/thickness mutations
                             and for flipping active status in grow/shrink.
        allow_air_gap:       If False (default), "Air" is excluded from
                             material mutations.

    Returns:
        (next_key, mutated_population)
    """
    # Build the set of allowed material indices for mutation
    if not allow_air_gap:
        # Exclude Air
        allowed_mats = jnp.array(
            [i for i, mat in enumerate(material_database) if mat.symbol != "Air"],
            dtype=jnp.int32,
        )
    else:
        allowed_mats = jnp.arange(len(material_database), dtype=jnp.int32)
    num_allowed_mats = allowed_mats.shape[0]

    pop_size, L = pop.materials.shape

    # Masks are identical for all individuals
    free_mat_mask = pop.free_material_mask[0]      # (L,)
    free_thick_mask = pop.free_thickness_mask[0]    # (L,)
    fully_free_mask = free_mat_mask & free_thick_mask  # only layers free in both aspects

    # Precompute indices of fully free layers (for scramble / grow-shrink)
    fully_free_indices = jnp.arange(L)[fully_free_mask]   # (num_fully_free,)
    num_fully_free = fully_free_indices.shape[0]

    # ------------------------------------------------------------------
    # 1.  Per‑individual subkeys
    # ------------------------------------------------------------------
    next_key, subkey = jax.random.split(key)
    indiv_keys = jax.random.split(subkey, pop_size)

    # ------------------------------------------------------------------
    # 2.  Per‑individual mutation
    # ------------------------------------------------------------------
    def _mutate_one(materials, thicknesses, active_mask, indiv_key):
        key_do, key_rest = jax.random.split(indiv_key)
        do_mutate = jax.random.bernoulli(key_do, mutation_rate)

        def no_mutation():
            return materials, thicknesses, active_mask

        type_key, key_s, key_g, key_m, key_t = jax.random.split(key_rest, 5)
        mutation_type = jax.random.randint(type_key, (), 0, 4)

        # ---- 0: Scramble free layers ----
        def _scramble(mats, thicks, acts):
            # Scramble only among fully free layers
            free_mats = mats[fully_free_mask]
            free_thicks = thicks[fully_free_mask]
            free_acts = acts[fully_free_mask]
            perm = jax.random.permutation(key_s, num_fully_free)
            mats = mats.at[fully_free_mask].set(free_mats[perm])
            thicks = thicks.at[fully_free_mask].set(free_thicks[perm])
            acts = acts.at[fully_free_mask].set(free_acts[perm])
            return mats, thicks, acts

        # ---- 1: Grow / Shrink ----
        def _grow_shrink(mats, thicks, acts):
            flip = jax.random.bernoulli(key_g, mutation_rate, shape=(L,))
            flip = flip & fully_free_mask   # only fully free layers can toggle
            new_acts = acts ^ flip

            # Ensure at least one fully free layer remains active
            num_active = jnp.sum(new_acts & fully_free_mask)
            chosen = jax.random.randint(key_g, (), 0, num_fully_free)
            layer_to_activate = fully_free_indices[chosen]
            new_acts = jnp.where(num_active == 0,
                                 new_acts.at[layer_to_activate].set(True),
                                 new_acts)
            return mats, thicks, new_acts

        # ---- 2: Material mutation (per free‑material layer) ----
        def _material_mutation(mats, thicks, acts):
            mut_mask = jax.random.bernoulli(key_m, mutation_rate, shape=(L,))
            mut_mask = mut_mask & free_mat_mask
            # Sample from allowed material indices
            rand_sub_idx = jax.random.randint(
                key_m, (L,), 0, num_allowed_mats, dtype=jnp.int32
            )
            new_mats = allowed_mats[rand_sub_idx].astype(mats.dtype)
            mats = jnp.where(mut_mask, new_mats, mats)
            return mats, thicks, acts

        # ---- 3: Thickness mutation (per free‑thickness layer) ----
        def _thickness_mutation(mats, thicks, acts):
            mut_mask = jax.random.bernoulli(key_t, mutation_rate, shape=(L,))
            mut_mask = mut_mask & free_thick_mask
            new_thicks_idx = jax.random.randint(
                key_t, (L,), 0, len(thickness_options), dtype=jnp.int32
            )
            new_thicks = thickness_options[new_thicks_idx]
            thicks = jnp.where(mut_mask, new_thicks, thicks)
            return mats, thicks, acts

        def apply_mutation():
            return lax.switch(
                mutation_type,
                [_scramble, _grow_shrink, _material_mutation, _thickness_mutation],
                materials, thicknesses, active_mask,
            )

        return lax.cond(do_mutate, apply_mutation, no_mutation)

    # ------------------------------------------------------------------
    # 3.  Vectorise over population
    # ------------------------------------------------------------------
    batched_fn = jax.vmap(_mutate_one, in_axes=(0, 0, 0, 0))
    new_mats, new_thicks, new_acts = batched_fn(
        pop.materials, pop.thicknesses_m, pop.active_mask, indiv_keys
    )

    # ------------------------------------------------------------------
    # 4.  Return mutated population
    # ------------------------------------------------------------------
    mutated_pop = Population(
        materials=new_mats,
        thicknesses_m=new_thicks,
        active_mask=new_acts,
        free_thickness_mask=pop.free_thickness_mask,
        free_material_mask=pop.free_material_mask,
    )

    return next_key, mutated_pop