"""
JaxFrame - A simple immutable DataFrame library
"""

from .dataframe import DataFrame
from .masked_array import MaskedArray
from .transform import (
    wide_to_long_masked, 
    long_to_wide_masked,
    wide_df_to_masked_array,
    masked_array_to_wide_df,
    roundtrip_wide_jax_conversion,
    pivot_sparse,
    unpivot_sparse,
    to_masked_array,
    from_masked_array,
    create_unpivot_skeleton,
)

__all__ = [
    "DataFrame",
    "MaskedArray",
    "wide_to_long_masked",
    "long_to_wide_masked",
    "wide_df_to_masked_array",
    "masked_array_to_wide_df",
    "pivot_sparse",
    "unpivot_sparse",
    "to_masked_array",
    "from_masked_array",
    "create_unpivot_skeleton",
]