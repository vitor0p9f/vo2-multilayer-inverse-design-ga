from ..types.structure import StructureTemplate, Mapping, Structure, Thicknesses, Materials, Mask
from typing import Dict, Optional, Tuple
from .expand_template import expand_template

import jax.numpy as jnp
import jax

def build_structure_from_template(
    template: StructureTemplate,
    max_layers: int,
    material_symbol_to_index: Dict[str, int],
    material_indices: Materials,
    thicknesses_um: Thicknesses,
    active_mask: Optional[Mask] = None,
) -> Structure:
    """
    Builds a concrete Structure from a template and explicit arrays.

    For layers that are fixed (material not free, thickness not free), the values
    from the template (e.g., "Si", jnp.inf) are used. For free layers, the provided
    arrays supply the values. The `active_mask` allows deactivating some of the
    free layers (wildcard expansions) to obtain variable effective length.

    Args:
        template: The structure template.
        max_layers: Max layers for wildcard expansion (must be consistent with mapping).
        material_symbol_to_index: Dict mapping symbol (e.g., "Si") to integer index.
        material_indices: Array of material indices for ALL layers (length must equal
            the number of layers in the expanded mapping).
        thicknesses_um: Array of thicknesses for ALL layers (same length).
        active_mask: Optional boolean mask indicating which layers are active
            (True = active, False = inactive/padding). If None, all layers are active.
            Incidence and substrate layers are forced to be active.

    Returns:
        Structure with merged fixed defaults, provided free values, and active mask.

    Raises:
        ValueError: If input arrays length mismatch or fixed values are missing.
    """
    # Step 1: Expand the template into a list of concrete Layer objects and the mapping
    expanded, mapping = expand_template(template, max_layers)
    n_layers = len(expanded)

    # Validate input array lengths
    if material_indices.shape[0] != n_layers:
        raise ValueError(
            f"material_indices length {material_indices.shape[0]} != mapping length {n_layers}"
        )
    if thicknesses_um.shape[0] != n_layers:
        raise ValueError(
            f"thicknesses_um length {thicknesses_um.shape[0]} != mapping length {n_layers}"
        )

    # Build free masks from mapping
    free_mat_mask = jnp.zeros(n_layers, dtype=bool)
    free_thick_mask = jnp.zeros(n_layers, dtype=bool)

    for i, m in enumerate(mapping):
        if m == Mapping.FREE_LAYER:
            # Both material and thickness are free
            free_mat_mask = free_mat_mask.at[i].set(True)
            free_thick_mask = free_thick_mask.at[i].set(True)
        elif m in (Mapping.PARTIAL_LAYER_MISSING_MATERIAL,
                   Mapping.PARTIAL_SUBSTRATE_MISSING_MATERIAL,
                   Mapping.PARTIAL_INCIDENCE_MEDIUM_MISSING_MATERIAL):
            # Only material is free, thickness is fixed
            free_mat_mask = free_mat_mask.at[i].set(True)
        elif m == Mapping.PARTIAL_LAYER_MISSING_THICKNESS:
            # Only thickness is free, material is fixed
            free_thick_mask = free_thick_mask.at[i].set(True)
        # For LAYER, SUBSTRATE, INCIDENCE_MEDIUM masks remain False (fixed)

    # Start with arrays filled with zeros (will be overwritten by fixed values)
    final_materials = jnp.zeros(n_layers, dtype=jnp.int8)
    final_thicknesses = jnp.zeros(n_layers, dtype=jnp.float32)

    # Fill fixed values from expanded layers
    for i, layer in enumerate(expanded):
        # Material
        if not free_mat_mask[i]:
            if layer.material_symbol is None:
                raise ValueError(f"Fixed material at position {i} has no symbol.")
            if layer.material_symbol not in material_symbol_to_index:
                raise KeyError(f"Unknown material symbol '{layer.material_symbol}'")
            final_materials = final_materials.at[i].set(
                material_symbol_to_index[layer.material_symbol]
            )
        # Thickness
        if not free_thick_mask[i]:
            if layer.thickness_um is None:
                raise ValueError(f"Fixed thickness at position {i} is None.")
            final_thicknesses = final_thicknesses.at[i].set(layer.thickness_um)

    # Overwrite free positions with provided arrays
    final_materials = jnp.where(free_mat_mask, material_indices, final_materials)
    final_thicknesses = jnp.where(free_thick_mask, thicknesses_um, final_thicknesses)

    # Handle active mask
    if active_mask is None:
        active_mask = jnp.ones(n_layers, dtype=bool)
    else:
        if active_mask.shape[0] != n_layers:
            raise ValueError(f"active_mask length {active_mask.shape[0]} != {n_layers}")
        # Force incidence and substrate layers to be always active
        inc_types = (Mapping.INCIDENCE_MEDIUM, Mapping.PARTIAL_INCIDENCE_MEDIUM_MISSING_MATERIAL)
        sub_types = (Mapping.SUBSTRATE, Mapping.PARTIAL_SUBSTRATE_MISSING_MATERIAL)
        for i, m in enumerate(mapping):
            if m in inc_types or m in sub_types:
                active_mask = active_mask.at[i].set(True)

    return Structure(
        materials=final_materials,
        thicknesses_um=final_thicknesses,
        active_mask=active_mask,
        free_thickness_mask=free_thick_mask,
        free_material_mask=free_mat_mask,
    )

def random_structure_from_template(
    template: StructureTemplate,
    max_layers: int,
    key: jax.random.PRNGKey,
    material_symbol_to_index: Dict[str, int],
    thickness_choices: jnp.ndarray,
) -> Tuple[jax.random.PRNGKey, Structure]:
    """
    Creates a random concrete Structure from a template, allowing variable effective length.
    Only FREE_LAYER positions (from wildcard expansion) may be deactivated; fixed layers
    (incidence, substrate, fixed layers) are always active. The number of active FREE_LAYERs
    is chosen uniformly between 0 and the total count of FREE_LAYER in the mapping.

    Args:
        template: Structure template.
        max_layers: Max layers for wildcard expansion.
        key: JAX PRNG key.
        material_symbol_to_index: Dict mapping symbol to integer index.
        thickness_choices: 1D array of allowed thickness values (e.g., [0.05, 0.1, 0.2]).

    Returns:
        A tuple (new_key, structure) where structure has an active_mask that defines
        the effective number of layers.
    """
    # First, expand the template to get the mapping and the number of layers
    expanded, mapping = expand_template(template, max_layers)
    n_layers = len(expanded)

    # Count how many FREE_LAYER positions exist (these come from wildcards or fixed free blocks)
    free_layer_count = sum(1 for m in mapping if m == Mapping.FREE_LAYER)

    # Split the key for independent random operations
    key_sub, key_mat, key_thick, key_act = jax.random.split(key, 4)

    # Decide randomly how many of the FREE_LAYER positions will be active
    num_active_free = jax.random.randint(key_act, (), 0, free_layer_count + 1)

    # Build active_mask: FREE_LAYER positions get True only if their index < num_active_free
    # (preserving order; the first `num_active_free` FREE_LAYERs are active, the rest inactive)
    active_mask = jnp.zeros(n_layers, dtype=bool)
    idx = 0
    for i, m in enumerate(mapping):
        if m == Mapping.FREE_LAYER:
            active_mask = active_mask.at[i].set(idx < num_active_free)
            idx += 1
        else:
            # Fixed layers (Incidence, Substrate, fixed Layer, etc.) are always active
            active_mask = active_mask.at[i].set(True)

    # Generate random material indices (uniform over number of materials)
    rand_mats = jax.random.randint(key_mat, (n_layers,), 0, len(material_symbol_to_index), dtype=jnp.int8)
    # Sample thicknesses uniformly from the provided choices
    thick_indices = jax.random.randint(key_thick, (n_layers,), 0, len(thickness_choices))
    rand_thicks = thickness_choices[thick_indices]

    # Build the concrete structure using the builder, passing the active mask
    struct = build_structure_from_template(
        template=template,
        max_layers=max_layers,
        material_symbol_to_index=material_symbol_to_index,
        material_indices=rand_mats,
        thicknesses_um=rand_thicks,
        active_mask=active_mask,
    )
    return key_sub, struct