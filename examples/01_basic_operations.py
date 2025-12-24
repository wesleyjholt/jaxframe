"""
Basic DataFrame Operations - JAXFrame Example

Shows fundamental DataFrame operations including:
- Creating DataFrames with mixed data types
- Column access and type preservation
- Immutable operations and chaining
- Basic data inspection
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import jax.numpy as jnp
from src.jaxframe import DataFrame

def main():
    print("=== Basic DataFrame Operations ===\n")
    
    # Create DataFrame with mixed types
    data = {
        'id': ['A', 'B', 'C', 'D'],
        'values': jnp.array([1.5, 2.3, 3.7, 4.1]),  # JAX array
        'scores': np.array([85, 92, 78, 96]),        # NumPy array  
        'categories': ['X', 'Y', 'X', 'Z']           # Python list
    }
    
    df = DataFrame(data, name="sample_data")
    print("1. DataFrame Creation:")
    print(df)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns}")
    print(f"Column types: {df.column_types}")
    print()
    
    # Column access preserves types
    print("2. Column Access (type preservation):")
    jax_col = df['values']
    numpy_col = df['scores'] 
    list_col = df['categories']
    
    print(f"JAX column type: {type(jax_col)} - shape: {jax_col.shape}")
    print(f"NumPy column type: {type(numpy_col)} - dtype: {numpy_col.dtype}")
    print(f"List column type: {type(list_col)} - length: {len(list_col)}")
    print()
    
    # Immutable operations
    print("3. Immutable Operations:")
    df_with_new_col = df.add_column('doubled', df['values'] * 2)
    print(f"Original DF columns: {df.columns}")
    print(f"New DF columns: {df_with_new_col.columns}")
    print(f"New column values: {list(df_with_new_col['doubled'])}")
    print()
    
    # DataFrame conversion
    print("4. Dictionary Conversion:")
    df_dict = df.to_dict()
    print("Keys:", list(df_dict.keys()))
    print("JAX array preserved:", type(df_dict['values']))
    print("NumPy array preserved:", type(df_dict['scores']))
    print()
    
    # Filtering example
    print("5. Basic Filtering:")
    high_scores = []
    high_ids = []
    for i, score in enumerate(df['scores']):
        if score > 85:
            high_scores.append(score)
            high_ids.append(df['id'][i])
    
    filtered_data = {'id': high_ids, 'scores': high_scores}
    filtered_df = DataFrame(filtered_data, name="high_scores")
    print("High scorers:")
    print(filtered_df)
    print()
    
    # Method chaining
    print("6. Method Chaining:")
    result = (df.add_column('rank', [1, 2, 3, 4])
               .with_name('ranked_data'))
    print(f"Final name: {result.name}")
    print(f"Final columns: {result.columns}")

if __name__ == "__main__":
    main()