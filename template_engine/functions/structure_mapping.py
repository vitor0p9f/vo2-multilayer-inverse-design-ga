from typing import Tuple, List, Union
from ..types.structure import StructureTemplate, FreeBlock, Mapping, Layer, Substrate, IncidenceMedium
from ..types.structure_mapping import StructureMapping

def _validate_incidence_substrate_counts(mapping: StructureMapping) -> None:
    """
    Validates that the final mapping contains exactly one incidence layer
    (complete or partial) and exactly one substrate layer (complete or partial).
    Raises ValueError otherwise.
    """
    incidence_types = (Mapping.INCIDENCE_MEDIUM, Mapping.PARTIAL_INCIDENCE_MEDIUM_MISSING_MATERIAL)
    substrate_types = (Mapping.SUBSTRATE, Mapping.PARTIAL_SUBSTRATE_MISSING_MATERIAL)
    inc_count = sum(1 for m in mapping if m in incidence_types)
    sub_count = sum(1 for m in mapping if m in substrate_types)
    errors = []
    if inc_count != 1:
        errors.append(f"Expected exactly 1 incidence layer, found {inc_count}.")
    if sub_count != 1:
        errors.append(f"Expected exactly 1 substrate layer, found {sub_count}.")
    if errors:
        raise ValueError("\n".join(errors))

def _reorder_mapping_and_items(
    mapping: List[Mapping],
    items: List[Union[Layer, FreeBlock]]
) -> Tuple[List[Mapping], List[Union[Layer, FreeBlock]]]:
    """
    Reorders the mapping and the corresponding list of template items so that:
    - Incidence entries (complete or partial) come first.
    - Substrate entries (complete or partial) come last.
    - All other entries keep their original relative order.

    This ensures that the final mapping has the required order: incidence first,
    then other layers/free blocks, then substrate last.
    """
    incidence_types = (Mapping.INCIDENCE_MEDIUM, Mapping.PARTIAL_INCIDENCE_MEDIUM_MISSING_MATERIAL)
    substrate_types = (Mapping.SUBSTRATE, Mapping.PARTIAL_SUBSTRATE_MISSING_MATERIAL)

    # Collect indices of incidence, substrate, and others.
    inc_indices = [i for i, m in enumerate(mapping) if m in incidence_types]
    sub_indices = [i for i, m in enumerate(mapping) if m in substrate_types]
    other_indices = [i for i in range(len(mapping)) if i not in inc_indices and i not in sub_indices]

    # Build the new order: incidence first, then others, then substrate.
    new_order = inc_indices + other_indices + sub_indices
    new_mapping = [mapping[i] for i in new_order]
    new_items = [items[i] for i in new_order]
    return new_mapping, new_items

def create_structure_mapping(template: StructureTemplate, max_layers: int) -> StructureMapping:
    """
    Expands the template into a fixed-length tuple of Mapping enums, with incidence first
    and substrate last. The expansion handles FreeBlock wildcards (number == 0) by distributing
    the remaining layers as evenly as possible among wildcards. Fixed FreeBlocks (number > 0)
    expand to that many FREE_LAYER entries. The function also validates that exactly one
    incidence and one substrate are present in the final mapping.
    """
    # ----- First pass: count fixed layers (non-wildcard) and collect wildcards -----
    fixed_layers = 0
    wildcards = []
    for item in template.layers:
        if isinstance(item, FreeBlock):
            if item.number < 0:
                raise ValueError(f"FreeBlock.number must be >= 0, got {item.number}.")
            if item.number > 0:
                fixed_layers += item.number   # fixed FreeBlock contributes its count
            else:
                wildcards.append(item)        # wildcard (number == 0)
        elif isinstance(item, (Substrate, IncidenceMedium, Layer)):
            fixed_layers += 1                 # each such item counts as one fixed layer
        else:
            raise TypeError(f"Unexpected item type: {type(item)}")

    if fixed_layers > max_layers:
        raise ValueError(f"Fixed layers ({fixed_layers}) exceed max_layers ({max_layers}).")

    remaining = max_layers - fixed_layers      # number of layers to distribute among wildcards
    num_wildcards = len(wildcards)

    # ----- Helper to build the mapping and item list in the original order -----
    def build_original():
        mapping = []
        items = []
        if num_wildcards == 0:
            # No wildcards: simply map each template item to its corresponding mapping type.
            for item in template.layers:
                if isinstance(item, FreeBlock) and item.number > 0:
                    mapping.extend([Mapping.FREE_LAYER] * item.number)
                    items.extend([item] * item.number)
                elif isinstance(item, Substrate):
                    mapping.append(Mapping.SUBSTRATE if item.material_symbol is not None else Mapping.PARTIAL_SUBSTRATE_MISSING_MATERIAL)
                    items.append(item)
                elif isinstance(item, IncidenceMedium):
                    mapping.append(Mapping.INCIDENCE_MEDIUM if item.material_symbol is not None else Mapping.PARTIAL_INCIDENCE_MEDIUM_MISSING_MATERIAL)
                    items.append(item)
                elif isinstance(item, Layer):
                    missing_mat = (item.material_symbol is None)
                    missing_thick = (item.thickness_um is None)
                    if missing_mat and missing_thick:
                        mapping.append(Mapping.FREE_LAYER)
                    elif missing_mat:
                        mapping.append(Mapping.PARTIAL_LAYER_MISSING_MATERIAL)
                    elif missing_thick:
                        mapping.append(Mapping.PARTIAL_LAYER_MISSING_THICKNESS)
                    else:
                        mapping.append(Mapping.LAYER)
                    items.append(item)
        else:
            # There are wildcards: distribute the remaining layers among them.
            if remaining < num_wildcards:
                raise ValueError(f"Remaining layers ({remaining}) insufficient for {num_wildcards} wildcards.")
            # Distribute as evenly as possible: base = floor(remaining / num_wildcards),
            # extra = remaining % num_wildcards → first 'extra' wildcards get base+1, others get base.
            base = remaining // num_wildcards
            extra = remaining % num_wildcards
            allocations = [base + 1] * extra + [base] * (num_wildcards - extra)
            wildcard_idx = 0
            for item in template.layers:
                if isinstance(item, FreeBlock):
                    if item.number > 0:
                        mapping.extend([Mapping.FREE_LAYER] * item.number)
                        items.extend([item] * item.number)
                    else:
                        # Wildcard: expand to the allocated number of FREE_LAYER entries.
                        alloc = allocations[wildcard_idx]
                        mapping.extend([Mapping.FREE_LAYER] * alloc)
                        items.extend([item] * alloc)
                        wildcard_idx += 1
                elif isinstance(item, Substrate):
                    mapping.append(Mapping.SUBSTRATE if item.material_symbol is not None else Mapping.PARTIAL_SUBSTRATE_MISSING_MATERIAL)
                    items.append(item)
                elif isinstance(item, IncidenceMedium):
                    mapping.append(Mapping.INCIDENCE_MEDIUM if item.material_symbol is not None else Mapping.PARTIAL_INCIDENCE_MEDIUM_MISSING_MATERIAL)
                    items.append(item)
                elif isinstance(item, Layer):
                    missing_mat = (item.material_symbol is None)
                    missing_thick = (item.thickness_um is None)
                    if missing_mat and missing_thick:
                        mapping.append(Mapping.FREE_LAYER)
                    elif missing_mat:
                        mapping.append(Mapping.PARTIAL_LAYER_MISSING_MATERIAL)
                    elif missing_thick:
                        mapping.append(Mapping.PARTIAL_LAYER_MISSING_THICKNESS)
                    else:
                        mapping.append(Mapping.LAYER)
                    items.append(item)
        return mapping, items

    # Build original (un-ordered) mapping and item list.
    mapping_original, items_original = build_original()

    # Reorder both mapping and items so that incidence entries come first and substrate last.
    mapping_ordered, items_ordered = _reorder_mapping_and_items(mapping_original, items_original)

    # Convert to tuple and validate the counts (exactly one incidence and one substrate).
    mapping_tuple = tuple(mapping_ordered)
    _validate_incidence_substrate_counts(mapping_tuple)

    # Return the ordered mapping. The reordered items are not needed further,
    # because expand_template will reorder the template itself using the same rule.
    return mapping_tuple