from typing import List, Tuple
from ..objects.template import Template
from specifications.objects.material import Material
from specifications.objects.structure import Structure
from specifications.types.structure import Thicknesses
from ..objects.layers import Mapping, FreeBlock, Layer

import jax
import jax.numpy as jnp

def build_random_structure_from_template(
    template: Template,
    key: jax.random.PRNGKey,
    materials: List[Material],
    thickness_options_m: Thicknesses,
    allow_air_gap: bool = False,
) -> Tuple[jax.random.PRNGKey, Structure]:
    """
    Creates a random concrete Structure from a template.

    Args:
        template: A validated Template (contains layer_mapping and max_layers).
        key: JAX PRNG key.
        materials: List of Material objects (used for index mapping). The
            material with symbol "Air" (if present) can be excluded from
            free layers via `allow_air_gap`.
        thickness_options_m: 1D array of allowed thicknesses in meters.
        allow_air_gap: If False (default), the material "Air" is excluded
            from random selection for free internal layers. The incidence
            medium and substrate are never affected by this flag — they
            use exactly the material specified in the template.

    Returns:
        (next_key, structure): The advanced PRNG key and a randomly
        generated Structure consistent with the template.
    """
    # ----- 1. Build symbol -> index mapping -----
    symbol_to_idx = {mat.symbol: i for i, mat in enumerate(materials)}
    num_materials = len(materials)

    # ----- 2. Expand the raw template into a list of Layer objects -----
    raw_items = template.layer_mapping
    # Separate incidence, substrate, middle
    incidence = raw_items[0]
    substrate = raw_items[-1]
    middle_raw = raw_items[1:-1]

    # Count fixed layers and collect wildcards (FreeBlock with number==0)
    fixed_layers = 0
    wildcards = []
    for item in middle_raw:
        if isinstance(item, FreeBlock):
            if item.number < 0:
                raise ValueError(f"FreeBlock.number must be >= 0, got {item.number}")
            if item.number > 0:
                fixed_layers += item.number
            else:
                wildcards.append(item)
        elif isinstance(item, Layer):
            fixed_layers += 1
        else:
            raise TypeError(f"Unexpected middle element type: {type(item)}")

    if fixed_layers > template.max_layers:
        raise ValueError(f"Fixed layers ({fixed_layers}) exceed max_layers ({template.max_layers})")

    remaining = template.max_layers - fixed_layers
    num_wildcards = len(wildcards)

    # Expand the middle part
    expanded_middle = []
    if num_wildcards == 0:
        for item in middle_raw:
            if isinstance(item, FreeBlock) and item.number > 0:
                for _ in range(item.number):
                    expanded_middle.append(Layer(material_symbol=None, thickness_m=None))
            elif isinstance(item, Layer):
                expanded_middle.append(item)
    else:
        if remaining < num_wildcards:
            raise ValueError(f"Remaining layers ({remaining}) insufficient for {num_wildcards} wildcards")
        base = remaining // num_wildcards
        extra = remaining % num_wildcards
        allocations = [base + 1] * extra + [base] * (num_wildcards - extra)
        wildcard_idx = 0
        for item in middle_raw:
            if isinstance(item, FreeBlock):
                if item.number > 0:
                    for _ in range(item.number):
                        expanded_middle.append(Layer(material_symbol=None, thickness_m=None))
                else:
                    alloc = allocations[wildcard_idx]
                    for _ in range(alloc):
                        expanded_middle.append(Layer(material_symbol=None, thickness_m=None))
                    wildcard_idx += 1
            elif isinstance(item, Layer):
                expanded_middle.append(item)

    # Combine incidence, expanded middle, substrate
    expanded_layers = [incidence] + expanded_middle + [substrate]
    n_layers = len(expanded_layers)

    mapping = template.expanded_mapping
    if len(mapping) != n_layers:
        raise RuntimeError("Expanded layer count does not match expanded mapping length")

    # ----- 3. Determine free masks based on mapping -----
    free_mat_mask = jnp.zeros(n_layers, dtype=bool)
    free_thick_mask = jnp.zeros(n_layers, dtype=bool)

    for i, m in enumerate(mapping):
        if m == Mapping.FREE_LAYER:
            free_mat_mask = free_mat_mask.at[i].set(True)
            free_thick_mask = free_thick_mask.at[i].set(True)
        elif m in (Mapping.PARTIAL_LAYER_MISSING_MATERIAL,
                   Mapping.PARTIAL_SUBSTRATE_MISSING_MATERIAL,
                   Mapping.PARTIAL_INCIDENCE_MEDIUM_MISSING_MATERIAL):
            free_mat_mask = free_mat_mask.at[i].set(True)
        elif m == Mapping.PARTIAL_LAYER_MISSING_THICKNESS:
            free_thick_mask = free_thick_mask.at[i].set(True)

    # Count FREE_LAYER positions for active mask
    free_layer_count = sum(1 for m in mapping if m == Mapping.FREE_LAYER)

    # ----- 4. Randomly decide active free layers -----
    next_key, key_mat, key_thick, key_act = jax.random.split(key, 4)

    num_active_free = jax.random.randint(key_act, (), 0, free_layer_count + 1)

    active_mask = jnp.ones(n_layers, dtype=bool)
    free_idx = 0
    for i, m in enumerate(mapping):
        if m == Mapping.FREE_LAYER:
            active_mask = active_mask.at[i].set(free_idx < num_active_free)
            free_idx += 1

    # ----- 5. Generate random materials (respecting allow_air_gap) -----
    if not allow_air_gap and "Air" in symbol_to_idx:
        air_idx = symbol_to_idx["Air"]
        allowed_list = [i for i in range(num_materials) if i != air_idx]
        allowed_indices = jnp.array(allowed_list, dtype=jnp.int8)
        # Random indices into allowed_indices
        rand_sub_idx = jax.random.randint(key_mat, (n_layers,), 0, len(allowed_list), dtype=jnp.int32)
        rand_mats = allowed_indices[rand_sub_idx]
    else:
        rand_mats = jax.random.randint(key_mat, (n_layers,), 0, num_materials, dtype=jnp.int8)

    thick_indices = jax.random.randint(key_thick, (n_layers,), 0, len(thickness_options_m))
    rand_thicks = thickness_options_m[thick_indices]

    # ----- 6. Fill fixed values from expanded layers, then overwrite free positions -----
    final_materials = jnp.zeros(n_layers, dtype=jnp.int8)
    final_thicknesses = jnp.zeros(n_layers, dtype=jnp.float32)

    for i, layer in enumerate(expanded_layers):
        if not free_mat_mask[i]:
            if layer.material_symbol is None:
                raise ValueError(f"Fixed material at position {i} has no symbol.")
            if layer.material_symbol not in symbol_to_idx:
                raise KeyError(f"Unknown material symbol '{layer.material_symbol}'")
            final_materials = final_materials.at[i].set(symbol_to_idx[layer.material_symbol])
        if not free_thick_mask[i]:
            if layer.thickness_m is None:
                raise ValueError(f"Fixed thickness at position {i} is None.")
            final_thicknesses = final_thicknesses.at[i].set(layer.thickness_m)

    # Overwrite free positions with random values
    final_materials = jnp.where(free_mat_mask, rand_mats, final_materials)
    final_thicknesses = jnp.where(free_thick_mask, rand_thicks, final_thicknesses)

    # Force incidence and substrate to be active
    inc_types = (Mapping.INCIDENCE_MEDIUM, Mapping.PARTIAL_INCIDENCE_MEDIUM_MISSING_MATERIAL)
    sub_types = (Mapping.SUBSTRATE, Mapping.PARTIAL_SUBSTRATE_MISSING_MATERIAL)
    for i, m in enumerate(mapping):
        if m in inc_types or m in sub_types:
            active_mask = active_mask.at[i].set(True)

    # ----- 7. Build and return Structure -----
    structure = Structure(
        materials=final_materials,
        thicknesses_m=final_thicknesses,
        active_mask=active_mask,
        free_thickness_mask=free_thick_mask,
        free_material_mask=free_mat_mask,
    )

    return next_key, structure