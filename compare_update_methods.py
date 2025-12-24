#!/usr/bin/env python3
"""
Compare Polars update() method with JAXFrame update_lookup_table() method.
"""

import polars as pl
import numpy as np
from src.jaxframe import DataFrame

def test_polars_update():
    print("=== POLARS UPDATE METHOD ===\n")
    
    # Example 1: Basic update by row index (default)
    print("1. Basic update by row index:")
    df = pl.DataFrame({
        "A": [1, 2, 3, 4],
        "B": [400, 500, 600, 700],
    })
    print(f"Original df:\n{df}\n")
    
    new_df = pl.DataFrame({
        "B": [-66, None, -99],
        "C": [5, 3, 1],
    })
    print(f"Update df:\n{new_df}\n")
    
    # Update by row index (Polars default)
    result = df.update(new_df)
    print(f"Result of df.update(new_df):\n{result}\n")
    print("Note: Only updates existing columns (B), ignores new columns (C), uses row index matching\n")
    
    # Example 2: Update with explicit join columns
    print("2. Update with explicit join columns:")
    df2 = pl.DataFrame({
        "id": [1, 2, 3, 4],
        "value": [100, 200, 300, 400],
        "name": ["Alice", "Bob", "Charlie", "David"]
    })
    print(f"Original df2:\n{df2}\n")
    
    update_df2 = pl.DataFrame({
        "id": [2, 4, 5],
        "value": [999, 777, 555],
        "status": ["updated", "modified", "new"]
    })
    print(f"Update df2:\n{update_df2}\n")
    
    # Update using 'id' column as join key
    result2 = df2.update(update_df2, on="id")
    print(f"Result of df2.update(update_df2, on='id'):\n{result2}\n")
    print("Note: Updates existing rows where id matches, ignores new columns\n")
    
    # Example 3: Full join (adds new rows)
    print("3. Update with how='full' (adds new rows):")
    result3 = df2.update(update_df2, on="id", how="full")
    print(f"Result with how='full':\n{result3}\n")
    print("Note: Adds new rows from update_df2 that don't exist in original\n")

def test_jaxframe_update_lookup_table():
    print("=== JAXFRAME UPDATE_LOOKUP_TABLE METHOD ===\n")
    
    # Example 1: Basic lookup table update
    print("1. Basic lookup table update:")
    df = DataFrame({
        "id": [1, 2, 3, 4],
        "value": [100, 200, 300, 400],
        "name": ["Alice", "Bob", "Charlie", "David"]
    })
    print(f"Original df:\n{df}\n")
    
    update_df = DataFrame({
        "id": [2, 4, 5],
        "value": [999, 777, 555],
        "name": ["Bob_Updated", "David_Updated", "Eve"]
    })
    print(f"Update df:\n{update_df}\n")
    
    # Update using 'id' as lookup key
    result = df.update_lookup_table(update_df, "id", strict=False)
    print(f"Result of df.update_lookup_table(update_df, 'id', strict=False):\n{result}\n")
    print("Note: Updates existing rows AND adds new rows, requires same column structure\n")
    
    # Example 2: Strict mode (error on conflicts)
    print("2. Strict mode with conflicts:")
    conflict_df = DataFrame({
        "id": [2, 3],
        "value": [999, 350],  # 350 conflicts with existing 300
        "name": ["Bob_Updated", "Charlie"]  # Charlie conflicts with existing Charlie
    })
    print(f"Conflict df:\n{conflict_df}\n")
    
    try:
        result_strict = df.update_lookup_table(conflict_df, "id", strict=True)
        print(f"Strict result:\n{result_strict}\n")
    except ValueError as e:
        print(f"Error in strict mode: {e}\n")
    
    # Example 3: Multi-column keys
    print("3. Multi-column lookup keys:")
    df_multi = DataFrame({
        "region": ["US", "US", "EU", "EU"],
        "product": ["A", "B", "A", "B"], 
        "sales": [100, 200, 150, 250]
    })
    print(f"Original df_multi:\n{df_multi}\n")
    
    update_multi = DataFrame({
        "region": ["US", "EU", "ASIA"],
        "product": ["A", "C", "A"],
        "sales": [999, 777, 300]
    })
    print(f"Update df_multi:\n{update_multi}\n")
    
    result_multi = df_multi.update_lookup_table(update_multi, ["region", "product"], strict=False)
    print(f"Result with multi-column key:\n{result_multi}\n")

def test_key_differences():
    print("=== KEY DIFFERENCES SUMMARY ===\n")
    
    # Demonstrate the core differences
    print("1. JOIN BEHAVIOR:")
    print("   - Polars update(): Updates by row index by default, or by specified join columns")
    print("   - JAXFrame update_lookup_table(): Always requires explicit id/key columns\n")
    
    print("2. COLUMN HANDLING:")
    print("   - Polars update(): Only updates existing columns, ignores new columns from update DataFrame")
    print("   - JAXFrame update_lookup_table(): Requires both DataFrames to have identical columns\n")
    
    print("3. NEW ROWS:")
    print("   - Polars update(): Adds new rows only with how='full'")
    print("   - JAXFrame update_lookup_table(): Always adds new rows (upsert behavior)\n")
    
    print("4. CONFLICT HANDLING:")
    print("   - Polars update(): Always overwrites, has include_nulls parameter")
    print("   - JAXFrame update_lookup_table(): Has strict mode to detect/prevent conflicts\n")
    
    print("5. VALIDATION:")
    print("   - Polars update(): No validation of key uniqueness")
    print("   - JAXFrame update_lookup_table(): Validates both DataFrames are valid lookup tables\n")

if __name__ == "__main__":
    try:
        test_polars_update()
    except ImportError:
        print("Polars not available, skipping Polars tests\n")
    
    test_jaxframe_update_lookup_table()
    test_key_differences()