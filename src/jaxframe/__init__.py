"""
JaxFrame - A simple immutable DataFrame library
"""

from .dataframe import DataFrame
from .transform import (
    wide_to_long_masked, 
    long_to_wide_masked,
    wide_df_to_jax_arrays,
    jax_arrays_to_wide_df,
    roundtrip_wide_jax_conversion
)

__all__ = ["DataFrame", "wide_to_long_masked", "long_to_wide_masked"]