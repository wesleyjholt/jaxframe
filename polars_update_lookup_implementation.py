#!/usr/bin/env python3
"""
Implementing JAXFrame's update_lookup_table using Polars operations.
"""

import polars as pl
import numpy as np
from src.jaxframe import DataFrame

def polars_update_lookup_table_strict_false(df_pl, other_pl, id_col):
    """
    Implement JAXFrame's update_lookup_table(strict=False) using Polars.
    This is equivalent to UPSERT with replacement.
    """
    # Step 1: Remove rows from df that have matching keys in other
    # This is like an anti-join to get rows that won't be updated
    non_matching = df_pl.join(other_pl, on=id_col, how="anti")
    
    # Step 2: Concatenate the non-matching rows with all rows from other
    # This gives us the UPSERT behavior
    result = pl.concat([non_matching, other_pl], how="vertical")
    
    # Step 3: Sort by the ID column to maintain consistent ordering
    result = result.sort(id_col)
    
    return result

def polars_update_lookup_table_strict_true(df_pl, other_pl, id_col):
    """
    Implement JAXFrame's update_lookup_table(strict=True) using Polars.
    This detects conflicts and raises errors.
    """
    # Step 1: Find overlapping keys
    overlapping_keys = df_pl.join(other_pl, on=id_col, how="inner")
    
    if len(overlapping_keys) > 0:
        # Step 2: Check for value conflicts in non-key columns
        # Join both dataframes and compare all non-key columns
        joined = df_pl.join(other_pl, on=id_col, how="inner", suffix="_other")
        
        # Get non-key columns
        non_key_cols = [col for col in df_pl.columns if col != id_col]
        
        # Check for conflicts
        for col in non_key_cols:
            left_col = col
            right_col = f"{col}_other"
            
            if right_col in joined.columns:
                # Check if any values differ
                conflicts = joined.filter(pl.col(left_col) != pl.col(right_col))
                
                if len(conflicts) > 0:
                    # Get the first conflict for error message
                    first_conflict = conflicts.row(0, named=True)
                    key_val = first_conflict[id_col]
                    existing_val = first_conflict[left_col]
                    new_val = first_conflict[right_col]
                    
                    raise ValueError(f"Value mismatch for key ({key_val},) in column '{col}': "
                                   f"existing='{existing_val}', new='{new_val}'")
    
    # If no conflicts, proceed with the upsert
    return polars_update_lookup_table_strict_false(df_pl, other_pl, id_col)

def test_polars_implementation():
    print("=== TESTING POLARS IMPLEMENTATION OF UPDATE_LOOKUP_TABLE ===\n")
    
    # Test data
    df_pl = pl.DataFrame({
        "id": [1, 2, 3, 4],
        "value": [100, 200, 300, 400],
        "name": ["Alice", "Bob", "Charlie", "David"]
    })
    
    update_pl = pl.DataFrame({
        "id": [2, 4, 5],
        "value": [999, 777, 555],
        "name": ["Bob_Updated", "David_Updated", "Eve"]
    })
    
    print("Original DataFrame:")
    print(df_pl)
    print("\nUpdate DataFrame:")
    print(update_pl)
    
    # Test strict=False (replacement)
    print("\n1. Testing strict=False (UPSERT with replacement):")
    result_false = polars_update_lookup_table_strict_false(df_pl, update_pl, "id")
    print(result_false)
    
    # Test strict=True with no conflicts
    print("\n2. Testing strict=True with no conflicts:")
    no_conflict_update = pl.DataFrame({
        "id": [2, 4, 5],
        "value": [200, 400, 555],  # Same values as original
        "name": ["Bob", "David", "Eve"]  # Same values as original
    })
    
    try:
        result_true = polars_update_lookup_table_strict_true(df_pl, no_conflict_update, "id")
        print(result_true)
    except ValueError as e:
        print(f"Error: {e}")
    
    # Test strict=True with conflicts
    print("\n3. Testing strict=True with conflicts:")
    conflict_update = pl.DataFrame({
        "id": [2, 4],
        "value": [999, 400],  # 999 conflicts with existing 200
        "name": ["Bob", "David"]
    })
    
    try:
        result_conflict = polars_update_lookup_table_strict_true(df_pl, conflict_update, "id")
        print(result_conflict)
    except ValueError as e:
        print(f"Error detected: {e}")

def compare_with_jaxframe():
    print("\n=== COMPARING WITH JAXFRAME IMPLEMENTATION ===\n")
    
    # JAXFrame version
    df_jax = DataFrame({
        "id": [1, 2, 3, 4],
        "value": [100, 200, 300, 400],
        "name": ["Alice", "Bob", "Charlie", "David"]
    })
    
    update_jax = DataFrame({
        "id": [2, 4, 5],
        "value": [999, 777, 555],
        "name": ["Bob_Updated", "David_Updated", "Eve"]
    })
    
    print("JAXFrame strict=False result:")
    jax_result = df_jax.update_lookup_table(update_jax, "id", strict=False)
    print(jax_result)
    
    # Polars version
    df_pl = pl.DataFrame({
        "id": [1, 2, 3, 4],
        "value": [100, 200, 300, 400],
        "name": ["Alice", "Bob", "Charlie", "David"]
    })
    
    update_pl = pl.DataFrame({
        "id": [2, 4, 5],
        "value": [999, 777, 555],
        "name": ["Bob_Updated", "David_Updated", "Eve"]
    })
    
    print("\nPolars equivalent result:")
    pl_result = polars_update_lookup_table_strict_false(df_pl, update_pl, "id")
    print(pl_result)
    
    print("\nNote: Results should be equivalent, though ordering might differ")

def advanced_polars_implementation():
    """
    More sophisticated Polars implementation that handles multi-column keys.
    """
    print("\n=== ADVANCED POLARS IMPLEMENTATION (Multi-column keys) ===\n")
    
    def polars_update_lookup_table_multi_key(df_pl, other_pl, id_cols, strict=False):
        """Handle multiple key columns"""
        if isinstance(id_cols, str):
            id_cols = [id_cols]
        
        if strict:
            # Check for conflicts in overlapping rows
            overlapping = df_pl.join(other_pl, on=id_cols, how="inner")
            
            if len(overlapping) > 0:
                joined = df_pl.join(other_pl, on=id_cols, how="inner", suffix="_other")
                non_key_cols = [col for col in df_pl.columns if col not in id_cols]
                
                for col in non_key_cols:
                    right_col = f"{col}_other"
                    if right_col in joined.columns:
                        conflicts = joined.filter(pl.col(col) != pl.col(right_col))
                        if len(conflicts) > 0:
                            first_conflict = conflicts.row(0, named=True)
                            key_vals = tuple(first_conflict[k] for k in id_cols)
                            raise ValueError(f"Value mismatch for key {key_vals} in column '{col}'")
        
        # Perform the upsert
        non_matching = df_pl.join(other_pl, on=id_cols, how="anti")
        result = pl.concat([non_matching, other_pl], how="vertical")
        result = result.sort(id_cols)
        
        return result
    
    # Test with multi-column keys
    df_multi = pl.DataFrame({
        "region": ["US", "US", "EU", "EU"],
        "product": ["A", "B", "A", "B"],
        "sales": [100, 200, 150, 250]
    })
    
    update_multi = pl.DataFrame({
        "region": ["US", "EU", "ASIA"],
        "product": ["A", "C", "A"],
        "sales": [999, 777, 300]
    })
    
    print("Multi-column key example:")
    print("Original:")
    print(df_multi)
    print("\nUpdate:")
    print(update_multi)
    print("\nResult:")
    result = polars_update_lookup_table_multi_key(df_multi, update_multi, ["region", "product"])
    print(result)

if __name__ == "__main__":
    test_polars_implementation()
    compare_with_jaxframe()
    advanced_polars_implementation()