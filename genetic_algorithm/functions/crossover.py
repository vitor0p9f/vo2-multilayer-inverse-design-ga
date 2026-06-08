from typing import Tuple, List
import jax
import jax.numpy as jnp
from jax import lax
from specifications.objects.structure import Structure

def crossover_structures(
    parent_pairs: List[Tuple[Structure, Structure]],
    key: jax.Array,
    n_cuts: int = 1,
    crossover_rate: float = 0.9,
    num_children_per_pair: int = 2,
) -> Tuple[jax.Array, List[Structure]]:
    """
    Multi‑point crossover for a batch of parent pairs.

    Args:
        parent_pairs:           List of (parent1, parent2) tuples.
        key:                    JAX PRNG key.
        n_cuts:                 Number of crossover points per pair.
        crossover_rate:         Probability a given pair undergoes crossover.
        num_children_per_pair:  Number of children per pair (1 or 2).

    Returns:
        (next_key, offspring_list)
    """
    if not parent_pairs:
        return key, []

    if num_children_per_pair not in (1, 2):
        raise ValueError("num_children_per_pair must be 1 or 2.")

    P = len(parent_pairs)

    # All parents share the same free masks
    ref = parent_pairs[0][0]
    free_mat_mask = ref.free_material_mask      # (L,)
    free_thick_mask = ref.free_thickness_mask    # (L,)

    # ------------------------------------------------------------------
    # 1.  Convert list of Structure pairs into batched arrays
    # ------------------------------------------------------------------
    mats_1   = jnp.stack([p[0].materials for p in parent_pairs])
    thick_1  = jnp.stack([p[0].thicknesses_m for p in parent_pairs])
    active_1 = jnp.stack([p[0].active_mask for p in parent_pairs])
    mats_2   = jnp.stack([p[1].materials for p in parent_pairs])
    thick_2  = jnp.stack([p[1].thicknesses_m for p in parent_pairs])
    active_2 = jnp.stack([p[1].active_mask for p in parent_pairs])

    # ------------------------------------------------------------------
    # 2.  Split keys for each pair
    # ------------------------------------------------------------------
    keys = jax.random.split(key, P + 1)
    next_key = keys[0]
    pair_keys = keys[1:]

    # ------------------------------------------------------------------
    # 3.  Per‑pair crossover logic
    # ------------------------------------------------------------------
    def _crossover_one_pair(mat1, thick1, act1, mat2, thick2, act2, pair_key):
        L_local = mat1.shape[0]

        cross_key, cut_key = jax.random.split(pair_key)
        do_cross = jax.random.bernoulli(cross_key, crossover_rate)

        def crossover_masks():
            # Maximum possible cuts, clamped to L_local - 1
            max_cuts = jnp.minimum(n_cuts, L_local - 1)

            # Random permutation of all possible cut positions 1..L-1
            all_pos = jnp.arange(1, L_local)
            perm = jax.random.permutation(cut_key, all_pos)

            # Select the first max_cuts elements using a boolean mask
            mask = jnp.arange(L_local - 1) < max_cuts

            # Scatter selected cuts into an indicator array (adds 1 at chosen positions)
            indicator = jnp.zeros(L_local, dtype=jnp.int32)
            indicator = indicator.at[perm].add(jnp.where(mask, 1, 0))

            # Build segment IDs by cumulative sum
            seg_id = jnp.cumsum(indicator)

            mask1 = (seg_id % 2 == 0)
            mask2 = ~mask1
            return mask1, mask2

        def no_crossover_masks():
            return jnp.ones(L_local, dtype=bool), jnp.zeros(L_local, dtype=bool)

        mask1, mask2 = lax.cond(do_cross, crossover_masks, no_crossover_masks)

        free_active = free_mat_mask | free_thick_mask

        # Materials (swap only where free_mat_mask is True)
        c1_mat = jnp.where(mask1 & free_mat_mask, mat1, mat2)
        c2_mat = jnp.where(mask2 & free_mat_mask, mat1, mat2)

        # Thicknesses (swap only where free_thick_mask is True)
        c1_thick = jnp.where(mask1 & free_thick_mask, thick1, thick2)
        c2_thick = jnp.where(mask2 & free_thick_mask, thick1, thick2)

        # Active flag (swap for any free layer)
        c1_act = jnp.where(mask1 & free_active, act1, act2)
        c2_act = jnp.where(mask2 & free_active, act1, act2)

        return c1_mat, c1_thick, c1_act, c2_mat, c2_thick, c2_act

    # ------------------------------------------------------------------
    # 4.  Vectorise over the batch
    # ------------------------------------------------------------------
    vmap_fn = jax.vmap(_crossover_one_pair, in_axes=(0, 0, 0, 0, 0, 0, 0))
    (c1_mats, c1_thicks, c1_acts,
     c2_mats, c2_thicks, c2_acts) = vmap_fn(mats_1, thick_1, active_1,
                                            mats_2, thick_2, active_2,
                                            pair_keys)

    # ------------------------------------------------------------------
    # 5.  Reconstruct offspring Structures
    # ------------------------------------------------------------------
    offspring = []
    for i in range(P):
        offspring.append(Structure(
            materials=c1_mats[i],
            thicknesses_m=c1_thicks[i],
            active_mask=c1_acts[i],
            free_thickness_mask=free_thick_mask,
            free_material_mask=free_mat_mask,
        ))
        if num_children_per_pair == 2:
            offspring.append(Structure(
                materials=c2_mats[i],
                thicknesses_m=c2_thicks[i],
                active_mask=c2_acts[i],
                free_thickness_mask=free_thick_mask,
                free_material_mask=free_mat_mask,
            ))

    return next_key, offspring