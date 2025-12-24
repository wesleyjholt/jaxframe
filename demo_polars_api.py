#!/usr/bin/env python3
"""
Demo script showing the new Polars-compatible methods in JAXFrame.
"""

import jax.numpy as jnp
from src.jaxframe import DataFrame

def main():
    print("=== JAXFrame Polars API Compatibility Demo ===\n")
    
    # Create sample data
    print("1. Creating sample DataFrames...")
    df1 = DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'score': [85.5, 92.0, 78.5]
    }, name="students")
    
    df2 = DataFrame({
        'name': ['David', 'Eve'],
        'age': [28, 26],
        'score': [88.0, 95.5]
    }, name="new_students")
    
    print(f"Original DataFrame:\n{df1}\n")
    print(f"Additional students:\n{df2}\n")
    
    # Demonstrate vstack
    print("2. Vertical stacking (vstack)...")
    combined = df1.vstack(df2)
    print(f"After vstack:\n{combined}\n")
    
    # Demonstrate hstack
    print("3. Horizontal stacking (hstack)...")
    extra_cols = [
        jnp.array([100, 200, 150]),  # bonus points
        jnp.array([90, 95, 85])      # grade scores
    ]
    df_with_extras = df1.hstack(extra_cols)
    print(f"After hstack with arrays:\n{df_with_extras}\n")
    
    # Demonstrate with_columns
    print("4. Adding columns (with_columns)...")
    df_with_bonus = df1.with_columns(
        bonus=jnp.array([10, 20, 15]),
        grade_score=[90, 95, 85]
    )
    print(f"After with_columns:\n{df_with_bonus}\n")
    
    # Demonstrate drop
    print("5. Dropping columns (drop)...")
    df_no_score = df_with_bonus.drop(['score', 'bonus'])
    print(f"After dropping 'score' and 'bonus':\n{df_no_score}\n")
    
    # Demonstrate filter
    print("6. Filtering rows (filter)...")
    
    # Filter by constraint
    young_students = df1.filter(age=25)
    print(f"Students aged 25:\n{young_students}\n")
    
    # Filter by boolean mask
    import numpy as np
    scores = np.array(df1['score'])
    high_scorers = df1.filter(scores > 85.0)
    print(f"High scorers (>85):\n{high_scorers}\n")
    
    # Demonstrate method chaining
    print("7. Method chaining (Polars-style)...")
    result = (df1
        .with_columns(
            bonus=jnp.array([10, 20, 15]),
            is_senior=[0, 1, 1]  # 0 for junior, 1 for senior
        )
        .filter(is_senior=1)
        .drop('score')
    )
    print(f"Chained operations result:\n{result}\n")
    
    # Show immutability
    print("8. Demonstrating immutability...")
    print(f"Original df1 unchanged:\n{df1}\n")
    
    print("=== Demo Complete ===")

if __name__ == "__main__":
    main()