from typing import Tuple
from ..types.structure import StructureTemplate, FreeBlock, Mapping, Layer, Substrate, IncidenceMedium

def create_layer_mapping(template: StructureTemplate, max_layers: int) -> Tuple[Mapping, ...]:
    """
    Expands a layer template into a fixed-length tuple of Mapping enums.

    The function processes a template that contains a sequence of layers (Substrate,
    IncidenceMedium, Layer) and free blocks (FreeBlock). Free blocks with number > 0
    expand to a fixed number of FREE_LAYER mappings. Free blocks with number == 0
    act as wildcards: they absorb any remaining layers (up to max_layers) and each
    wildcard is replaced by one or more FREE_LAYER entries.
    When multiple wildcards are present, the remaining layers are distributed as
    evenly as possible while preserving the original order.

    Args:
        template (StructureTemplate): The template object containing a list of
            layers and free blocks. Each element can be a Substrate, IncidenceMedium,
            Layer, or FreeBlock.
        max_layers (int): The total number of layers after expansion. Must be
            greater than or equal to the number of fixed layers in the template.

    Returns:
        Tuple[Mapping, ...]: A tuple of Mapping enum values with length exactly
            equal to max_layers. Each value indicates the type and completeness
            of the corresponding layer in the final structure.

    Raises:
        ValueError: In the following cases:
            - A FreeBlock has a negative number.
            - The total number of fixed layers exceeds max_layers.
            - There are no wildcards and fixed_layers != max_layers.
            - There are wildcards but the remaining layers are fewer than the
              number of wildcards (each wildcard needs at least one layer).
        TypeError: If an item in template.layers is not one of the expected types.

    Example:
        >>> template = StructureTemplate(layers=[
        ...     FreeBlock(number=0),                           # wildcard 1
        ...     Layer(material_symbol=None, thickness_um=0.1),  # missing material
        ...     FreeBlock(number=2)                             # fixed free block
        ... ])
        >>> create_layer_mapping(template, max_layers=5)
        (Mapping.FREE_LAYER,                     # wildcard 1 → first of 2 layers
         Mapping.FREE_LAYER,                     # wildcard 1 → second of 2 layers
         Mapping.PARTIAL_LAYER_MISSING_MATERIAL, # Layer with missing material
         Mapping.FREE_LAYER,                     # FreeBlock(2) → first
         Mapping.FREE_LAYER)                     # FreeBlock(2) → second

    Notes:
        - Fixed layers are counted from:
            * FreeBlock with number > 0 (adds that many fixed layers)
            * Substrate (1 layer)
            * IncidenceMedium (1 layer)
            * Layer (1 layer)
        - Wildcards (FreeBlock with number == 0) are not counted as fixed.
        - Both wildcards and fixed FreeBlock (number > 0) expand to FREE_LAYER.
        - A Substrate with missing material (material_symbol is None) also becomes FREE_LAYER.
        - The distribution among wildcards uses a round‑robin style:
          remaining = max_layers - fixed_layers
          base = remaining // num_wildcards
          extra = remaining % num_wildcards
          The first `extra` wildcards get `base + 1` layers, the rest get `base`.
    """
    fixed_layers = 0
    wildcards = []

    for item in template.layers:
        if isinstance(item, FreeBlock):
            number = item.number
            if number < 0:
                raise ValueError(f"FreeBlock.number must be >= 0, got {number}.")
            if number > 0:
                fixed_layers += number
            else:
                wildcards.append(item)
        elif isinstance(item, (Substrate, IncidenceMedium, Layer)):
            fixed_layers += 1
        else:
            raise TypeError(f"Unexpected item type: {type(item)}")

    if fixed_layers > max_layers:
        raise ValueError(
            f"Fixed layers ({fixed_layers}) exceed max_layers ({max_layers})."
        )

    remaining = max_layers - fixed_layers
    num_wildcards = len(wildcards)

    if num_wildcards == 0:
        if remaining != 0:
            raise ValueError(
                f"Template without wildcards must have exactly {max_layers} layers, "
                f"but contains {fixed_layers}."
            )

        mapping = []
        for item in template.layers:
            if isinstance(item, FreeBlock) and item.number > 0:
                mapping.extend([Mapping.FREE_BLOCK] * item.number)
            elif isinstance(item, Substrate):
                mapping.append(
                    Mapping.SUBSTRATE if item.material_symbol is not None
                    else Mapping.FREE_LAYER
                )
            elif isinstance(item, IncidenceMedium):
                mapping.append(
                    Mapping.INCIDENCE_MEDIUM if item.material_symbol is not None
                    else Mapping.PARTIAL_INCIDENCE_MEDIUM_MISSING_MATERIAL
                )
            elif isinstance(item, Layer):
                missing_material = (item.material_symbol is None)
                missing_thickness = (item.thickness_um is None)

                if missing_material and missing_thickness:
                    mapping.append(Mapping.FREE_LAYER)
                elif missing_material:
                    mapping.append(Mapping.PARTIAL_LAYER_MISSING_MATERIAL)
                elif missing_thickness:
                    mapping.append(Mapping.PARTIAL_LAYER_MISSING_THICKNESS)
                else:
                    mapping.append(Mapping.LAYER)

        return tuple(mapping)

    if remaining < num_wildcards:
        raise ValueError(
            f"Remaining layers ({remaining}) are insufficient to distribute "
            f"among {num_wildcards} wildcard(s) (each needs at least one layer)."
        )

    base = remaining // num_wildcards
    extra = remaining % num_wildcards
    allocations = [base + 1] * extra + [base] * (num_wildcards - extra)

    mapping = []
    wildcard_index = 0

    for item in template.layers:
        if isinstance(item, FreeBlock):
            number = item.number
            if number > 0:
                mapping.extend([Mapping.FREE_LAYER] * number)
            else:
                alloc = allocations[wildcard_index]
                mapping.extend([Mapping.FREE_LAYER] * alloc)
                wildcard_index += 1
        elif isinstance(item, Substrate):
            mapping.append(
                Mapping.SUBSTRATE if item.material_symbol is not None
                else Mapping.FREE_LAYER
            )
        elif isinstance(item, IncidenceMedium):
            mapping.append(
                Mapping.INCIDENCE_MEDIUM if item.material_symbol is not None
                else Mapping.PARTIAL_INCIDENCE_MEDIUM_MISSING_MATERIAL
            )
        elif isinstance(item, Layer):
            missing_material = (item.material_symbol is None)
            missing_thickness = (item.thickness_um is None)

            if missing_material and missing_thickness:
                mapping.append(Mapping.FREE_LAYER)
            elif missing_material:
                mapping.append(Mapping.PARTIAL_LAYER_MISSING_MATERIAL)
            elif missing_thickness:
                mapping.append(Mapping.PARTIAL_LAYER_MISSING_THICKNESS)
            else:
                mapping.append(Mapping.LAYER)
        else:
            raise TypeError(f"Unexpected item type: {type(item)}")

    return tuple(mapping)