#!/usr/bin/env python3
"""
Additional edge case tests to highlight differences between Polars update and JAXFrame update_lookup_table.
"""

import polars as pl
import numpy as np
from src.jaxframe import DataFrame

def test_edge_cases():
    print("=== EDGE CASE COMPARISONS ===\n")
    
    # Edge Case 1: Different column sets
    print("1. DIFFERENT COLUMN SETS:")
    print("Polars approach:")
    df1 = pl.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35]
    })
    
    # Update with different columns
    update1 = pl.DataFrame({
        "id": [2, 4],
        "salary": [50000, 60000],  # New column
        "age": [31, 40]  # Existing column
    })
    
    result1 = df1.update(update1, on="id", how="full")
    print(f"Polars result (different columns):\n{result1}\n")
    print("Note: Polars ignores 'salary' column, only updates 'age'\n")
    
    print("JAXFrame approach:")
    df2 = DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35]
    })
    
    # Try to update with different columns - this will fail
    try:
        update2 = DataFrame({
            "id": [2, 4],
            "salary": [50000, 60000],
            "age": [31, 40]
        })
        result2 = df2.update_lookup_table(update2, "id", strict=False)
        print(f"JAXFrame result:\n{result2}\n")
    except ValueError as e:
        print(f"JAXFrame error: {e}\n")
    
    # Edge Case 2: Null/None handling
    print("2. NULL/NONE HANDLING:")
    print("Polars approach:")
    df3 = pl.DataFrame({
        "id": [1, 2, 3],
        "value": [10, 20, 30]
    })
    
    update3 = pl.DataFrame({
        "id": [2, 3],
        "value": [None, 35]
    })
    
    # Default behavior (ignore nulls)
    result3a = df3.update(update3, on="id")
    print(f"Polars result (ignore nulls):\n{result3a}\n")
    
    # Include nulls
    result3b = df3.update(update3, on="id", include_nulls=True)
    print(f"Polars result (include nulls):\n{result3b}\n")
    
    print("JAXFrame approach:")
    df4 = DataFrame({
        "id": [1, 2, 3],
        "value": [10, 20, 30]
    })
    
    # JAXFrame doesn't have built-in null handling - uses None in lists
    update4 = DataFrame({
        "id": [2, 3],
        "value": [None, 35]
    })
    
    result4 = df4.update_lookup_table(update4, "id", strict=False)
    print(f"JAXFrame result (None values):\n{result4}\n")
    
    # Edge Case 3: Duplicate keys in update data
    print("3. DUPLICATE KEYS IN UPDATE DATA:")
    print("Polars approach:")
    df5 = pl.DataFrame({
        "id": [1, 2, 3],
        "value": [10, 20, 30]
    })
    
    # Polars allows duplicates in update data
    update5 = pl.DataFrame({
        "id": [2, 2, 3],  # Duplicate id=2
        "value": [100, 200, 300]
    })
    
    try:
        result5 = df5.update(update5, on="id")
        print(f"Polars result (duplicate keys):\n{result5}\n")
        print("Note: Polars handles duplicates (likely uses last value)\n")
    except Exception as e:
        print(f"Polars error with duplicates: {e}\n")
    
    print("JAXFrame approach:")
    df6 = DataFrame({
        "id": [1, 2, 3],
        "value": [10, 20, 30]
    })
    
    try:
        update6 = DataFrame({
            "id": [2, 2, 3],  # Duplicate id=2
            "value": [100, 200, 300]
        })
        result6 = df6.update_lookup_table(update6, "id", strict=False)
        print(f"JAXFrame result:\n{result6}\n")
    except ValueError as e:
        print(f"JAXFrame error (duplicate keys): {e}\n")
    
    # Edge Case 4: Row index vs explicit join
    print("4. ROW INDEX vs EXPLICIT JOIN:")
    print("Polars default (row index):")
    df7 = pl.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "score": [85, 90, 75]
    })
    
    update7 = pl.DataFrame({
        "score": [95, 80],  # Only 2 rows
        "grade": ["A", "B"]
    })
    
    result7 = df7.update(update7)  # No 'on' parameter = row index join
    print(f"Polars row index update:\n{result7}\n")
    print("Note: Updates first 2 rows by position, ignores mismatched lengths\n")
    
    print("JAXFrame always requires explicit keys:")
    print("JAXFrame cannot do row index updates - always needs id_columns\n")

def test_performance_considerations():
    print("=== PERFORMANCE & DESIGN CONSIDERATIONS ===\n")
    
    print("1. LOOKUP TABLE VALIDATION:")
    print("   - Polars: No validation, allows duplicate keys in both tables")
    print("   - JAXFrame: Validates uniqueness, ensures referential integrity\n")
    
    print("2. SCHEMA FLEXIBILITY:")
    print("   - Polars: Flexible schema, can update subset of columns")
    print("   - JAXFrame: Strict schema matching, all columns must be present\n")
    
    print("3. OPERATION TYPE:")
    print("   - Polars: General-purpose update/merge operation")
    print("   - JAXFrame: Specialized lookup table maintenance (UPSERT)\n")
    
    print("4. ERROR HANDLING:")
    print("   - Polars: Permissive, handles edge cases gracefully")
    print("   - JAXFrame: Strict validation, fails fast on inconsistencies\n")

if __name__ == "__main__":
    test_edge_cases()
    test_performance_considerations()