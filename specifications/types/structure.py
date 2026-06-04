from jaxtyping import Int8, Float32, Bool, Array

Materials = Int8[Array, "n_layers"]
Thicknesses = Float32[Array, "n_layers"]
Mask = Bool[Array, "n_layers"]