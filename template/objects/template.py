from dataclasses import dataclass
from typing import List, Tuple, Union
from jax.tree_util import register_dataclass
from .layers import LayerMapping, Substrate, IncidenceMedium, FreeBlock, Mapping, Layer

@register_dataclass
@dataclass(frozen=True)
class Template:
    """
    A validated template that defines a sequence of optical layers.
    Enforces:
        - Must contain exactly one Substrate and one IncidenceMedium.
        - IncidenceMedium must be the first element in the raw template.
        - Substrate must be the last element in the raw template.
        - At least one actual layer (Layer or FreeBlock) exists between them.
        - Total number of expanded actual layers (middle) does not exceed max_layers.
        - max_layers refers ONLY to the number of actual layers (excluding incidence and substrate).
    """
    layer_mapping: LayerMapping   # sequence of Layer, Substrate, IncidenceMedium, FreeBlock
    max_layers: int               # maximum number of actual layers (between incidence and substrate)

    def __post_init__(self):
        self._validate_raw_order()
        expanded = self._expand_and_reorder()
        object.__setattr__(self, '_expanded_mapping', expanded)

    def _validate_raw_order(self):
        mapping = self.layer_mapping
        if len(mapping) < 3:
            raise ValueError("Template must have at least 3 elements: IncidenceMedium, at least one actual layer, Substrate")
        if not isinstance(mapping[0], IncidenceMedium):
            raise ValueError("First element must be IncidenceMedium")
        if not isinstance(mapping[-1], Substrate):
            raise ValueError("Last element must be Substrate")
        incidence_count = sum(1 for elem in mapping if isinstance(elem, IncidenceMedium))
        substrate_count = sum(1 for elem in mapping if isinstance(elem, Substrate))
        if incidence_count != 1:
            raise ValueError("Exactly one IncidenceMedium allowed")
        if substrate_count != 1:
            raise ValueError("Exactly one Substrate allowed")
        middle = mapping[1:-1]
        if len(middle) == 0:
            raise ValueError("At least one actual layer (Layer or FreeBlock) must exist between IncidenceMedium and Substrate")
        for elem in middle:
            if isinstance(elem, (IncidenceMedium, Substrate)):
                raise ValueError("IncidenceMedium or Substrate cannot appear in the middle")

    def _count_min_layers(self) -> int:
        """Minimum number of actual layers that are NOT free (i.e., Layer objects). FreeBlocks contribute 0."""
        middle = self.layer_mapping[1:-1]
        # Only Layer objects count; FreeBlocks (even with number>0) are free layers.
        return sum(1 for item in middle if isinstance(item, Layer))

    def _expand_and_reorder(self) -> Tuple[Mapping, ...]:
        """
        Expand FreeBlock wildcards (number == 0) to fill as many layers as possible
        up to max_layers, distribute remaining layers among wildcards (only for actual layers).
        Incidence and substrate are placed at the ends after expansion.
        Returns a tuple of Mapping enums (incidence, expanded layers, substrate).
        """
        items = self.layer_mapping
        # Separate incidence, substrate, and actual layers (middle)
        incidence = items[0]
        substrate = items[-1]
        middle = items[1:-1]

        # Count fixed layers from middle and collect wildcards
        fixed_layers = 0
        wildcards = []
        for item in middle:
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

        if fixed_layers > self.max_layers:
            raise ValueError(f"Fixed actual layers ({fixed_layers}) exceed max_layers ({self.max_layers})")

        remaining = self.max_layers - fixed_layers
        num_wildcards = len(wildcards)

        # Expand middle items into a list of Mapping enums
        expanded_middle = []
        if num_wildcards == 0:
            for item in middle:
                if isinstance(item, FreeBlock) and item.number > 0:
                    expanded_middle.extend([Mapping.FREE_LAYER] * item.number)
                elif isinstance(item, Layer):
                    missing_mat = item.material_symbol is None
                    missing_thick = item.thickness_m is None
                    if missing_mat and missing_thick:
                        expanded_middle.append(Mapping.FREE_LAYER)
                    elif missing_mat:
                        expanded_middle.append(Mapping.PARTIAL_LAYER_MISSING_MATERIAL)
                    elif missing_thick:
                        expanded_middle.append(Mapping.PARTIAL_LAYER_MISSING_THICKNESS)
                    else:
                        expanded_middle.append(Mapping.LAYER)
        else:
            if remaining < num_wildcards:
                raise ValueError(f"Remaining layers ({remaining}) insufficient for {num_wildcards} wildcards.")
            base = remaining // num_wildcards
            extra = remaining % num_wildcards
            allocations = [base + 1] * extra + [base] * (num_wildcards - extra)
            wildcard_idx = 0
            for item in middle:
                if isinstance(item, FreeBlock):
                    if item.number > 0:
                        expanded_middle.extend([Mapping.FREE_LAYER] * item.number)
                    else:
                        alloc = allocations[wildcard_idx]
                        expanded_middle.extend([Mapping.FREE_LAYER] * alloc)
                        wildcard_idx += 1
                elif isinstance(item, Layer):
                    missing_mat = item.material_symbol is None
                    missing_thick = item.thickness_m is None
                    if missing_mat and missing_thick:
                        expanded_middle.append(Mapping.FREE_LAYER)
                    elif missing_mat:
                        expanded_middle.append(Mapping.PARTIAL_LAYER_MISSING_MATERIAL)
                    elif missing_thick:
                        expanded_middle.append(Mapping.PARTIAL_LAYER_MISSING_THICKNESS)
                    else:
                        expanded_middle.append(Mapping.LAYER)

        # Build incidence and substrate mappings
        inc_mapping = (Mapping.INCIDENCE_MEDIUM if incidence.material_symbol is not None
                       else Mapping.PARTIAL_INCIDENCE_MEDIUM_MISSING_MATERIAL)
        sub_mapping = (Mapping.SUBSTRATE if substrate.material_symbol is not None
                       else Mapping.PARTIAL_SUBSTRATE_MISSING_MATERIAL)

        # Combine: incidence first, expanded middle, substrate last
        ordered = [inc_mapping] + expanded_middle + [sub_mapping]
        return tuple(ordered)

    @property
    def expanded_mapping(self) -> Tuple[Mapping, ...]:
        return self._expanded_mapping

    def is_complete(self) -> bool:
        """Return True if every element (including incidence and substrate) has all required attributes."""
        complete_states = {Mapping.LAYER, Mapping.SUBSTRATE, Mapping.INCIDENCE_MEDIUM, Mapping.FREE_LAYER}
        return all(m in complete_states for m in self.expanded_mapping)

    def summary(self) -> dict:
        """
        Return a detailed summary of the template.
        Incidence medium and substrate are NOT counted as "layers".
        Only the actual layers (middle elements) are considered in layer counts.
        """
        expanded = self.expanded_mapping
        # Convert enum to readable names
        expanded_names = tuple(m.name for m in expanded)

        # Separate incidence and substrate from actual layers
        incidence_types = (Mapping.INCIDENCE_MEDIUM, Mapping.PARTIAL_INCIDENCE_MEDIUM_MISSING_MATERIAL)
        substrate_types = (Mapping.SUBSTRATE, Mapping.PARTIAL_SUBSTRATE_MISSING_MATERIAL)

        actual_layers = [m for m in expanded if m not in incidence_types and m not in substrate_types]
        num_actual = len(actual_layers)

        free_actual = actual_layers.count(Mapping.FREE_LAYER)
        missing_mat_actual = sum(1 for m in actual_layers if m == Mapping.PARTIAL_LAYER_MISSING_MATERIAL)
        missing_thick_actual = actual_layers.count(Mapping.PARTIAL_LAYER_MISSING_THICKNESS)
        complete_actual = actual_layers.count(Mapping.LAYER)

        fixed_actual = num_actual - free_actual

        return {
            "min_layers": self._count_min_layers(),
            "max_layers": self.max_layers,
            "number_of_fixed_layers": fixed_actual,
            "number_of_free_layers": free_actual,
            "number_of_missing_material_layers": missing_mat_actual,
            "number_of_missing_thickness_layer": missing_thick_actual,
            "all_layers_are_complete": self.is_complete(),
            "expanded_mapping": expanded_names,   # now a tuple of strings
        }