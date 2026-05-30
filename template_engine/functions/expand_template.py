from ..types.structure import StructureTemplate, Layer, Mapping, FreeBlock, Substrate, IncidenceMedium
from typing import Tuple, List, Union
from .structure_mapping import create_structure_mapping
from ..types.structure_mapping import StructureMapping
import jax.numpy as jnp

def _reorder_template_layers(template: StructureTemplate) -> List[Union[Layer, FreeBlock]]:
    """
    Reorders template layers so that IncidenceMedium come first,
    Substrate come last, and others keep original order.
    """
    items = list(template.layers)
    inc_items = [i for i in items if isinstance(i, IncidenceMedium)]
    sub_items = [i for i in items if isinstance(i, Substrate)]
    other_items = [i for i in items if not isinstance(i, (IncidenceMedium, Substrate))]
    return inc_items + other_items + sub_items

def expand_template(
    template: StructureTemplate,
    max_layers: int,
) -> Tuple[List[Layer], StructureMapping]:
    """
    Expands the template into a concrete list of Layer objects (with default values)
    using the precomputed mapping. The mapping is already ordered (incidence first,
    substrate last), so the template layers are reordered internally to match.
    """
    # Get the ordered mapping (incidence -> others -> substrate) from the template.
    # This mapping already has length = max_layers, with wildcards expanded.
    mapping = create_structure_mapping(template, max_layers)

    # Reorder the template items to match the order of the mapping:
    # incidence first, then other items (layers, free blocks), substrate last.
    reordered_items = _reorder_template_layers(template)

    expanded = []          # will hold the concrete Layer objects
    map_idx = 0            # current position in the mapping tuple

    # Iterate over the reordered template items, consuming the mapping in order.
    for item in reordered_items:
        if isinstance(item, FreeBlock):
            if item.number > 0:
                # Fixed FreeBlock: expands to exactly `item.number` FREE_LAYER entries.
                for _ in range(item.number):
                    # Safety check: mapping must contain FREE_LAYER at these positions.
                    if mapping[map_idx] != Mapping.FREE_LAYER:
                        raise RuntimeError(f"Expected FREE_LAYER, got {mapping[map_idx]}")
                    expanded.append(Layer(material_symbol=None, thickness_um=None))
                    map_idx += 1
            else:
                # Wildcard FreeBlock (number == 0): expands to a consecutive block of FREE_LAYER.
                # Determine how many FREE_LAYER entries belong to this wildcard by counting
                # contiguous FREE_LAYER in the mapping starting at current index.
                count = 0
                while map_idx + count < len(mapping) and mapping[map_idx + count] == Mapping.FREE_LAYER:
                    count += 1
                for _ in range(count):
                    expanded.append(Layer(material_symbol=None, thickness_um=None))
                map_idx += count

        elif isinstance(item, Substrate):
            # Exactly one mapping entry per Substrate item.
            m = mapping[map_idx]
            if m == Mapping.SUBSTRATE:
                # Complete substrate with material symbol provided.
                expanded.append(Substrate(
                    material_symbol=item.material_symbol,
                    thickness_um=item.thickness_um if item.thickness_um is not None else jnp.inf
                ))
            elif m == Mapping.PARTIAL_SUBSTRATE_MISSING_MATERIAL:
                # Substrate with missing material (material will be optimized).
                expanded.append(Substrate(
                    material_symbol=None,
                    thickness_um=item.thickness_um if item.thickness_um is not None else jnp.inf
                ))
            elif m == Mapping.FREE_LAYER:
                # If mapping says FREE_LAYER, treat as a completely free layer (no fixed defaults).
                expanded.append(Layer(material_symbol=None, thickness_um=None))
            else:
                raise RuntimeError(f"Unexpected mapping {m} for Substrate")
            map_idx += 1

        elif isinstance(item, IncidenceMedium):
            # Exactly one mapping entry per IncidenceMedium item.
            m = mapping[map_idx]
            if m == Mapping.INCIDENCE_MEDIUM:
                expanded.append(IncidenceMedium(
                    material_symbol=item.material_symbol,
                    thickness_um=item.thickness_um if item.thickness_um is not None else jnp.inf
                ))
            elif m == Mapping.PARTIAL_INCIDENCE_MEDIUM_MISSING_MATERIAL:
                expanded.append(IncidenceMedium(material_symbol=None, thickness_um=jnp.inf))
            elif m == Mapping.FREE_LAYER:
                expanded.append(Layer(material_symbol=None, thickness_um=None))
            else:
                raise RuntimeError(f"Unexpected mapping {m} for IncidenceMedium")
            map_idx += 1

        elif isinstance(item, Layer):
            # Exactly one mapping entry per fixed Layer item.
            m = mapping[map_idx]
            if m == Mapping.LAYER:
                # Fully defined layer: both material and thickness are fixed.
                expanded.append(Layer(
                    material_symbol=item.material_symbol,
                    thickness_um=item.thickness_um
                ))
            elif m == Mapping.FREE_LAYER:
                # Totally free layer (no defaults from template).
                expanded.append(Layer(material_symbol=None, thickness_um=None))
            elif m == Mapping.PARTIAL_LAYER_MISSING_MATERIAL:
                # Material is free, thickness is fixed to template value.
                expanded.append(Layer(material_symbol=None, thickness_um=item.thickness_um))
            elif m == Mapping.PARTIAL_LAYER_MISSING_THICKNESS:
                # Thickness is free, material is fixed to template value.
                expanded.append(Layer(material_symbol=item.material_symbol, thickness_um=None))
            else:
                raise RuntimeError(f"Unexpected mapping {m} for Layer")
            map_idx += 1

        else:
            raise TypeError(f"Unexpected item type: {type(item)}")

    # After processing all items, map_idx should have consumed the entire mapping.
    assert map_idx == len(mapping), "Mapping not fully consumed"

    return expanded, mapping