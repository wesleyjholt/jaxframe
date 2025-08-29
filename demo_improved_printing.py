#!/usr/bin/env python3
"""
Demo showing the improved DataFrame printing functionality.
"""

from src.jaxframe import DataFrame
import numpy as np

def main():
    print("=" * 60)
    print("JAXFRAME DATAFRAME - IMPROVED PRINTING DEMO")
    print("=" * 60)
    print()
    
    # Create a sample dataset similar to the user's example
    print("Creating sample DataFrame with mixed data types...")
    data = {
        'volume_fraction': ['0.108', '0.205', '0.142', '0.067', '0.234'],
        'pore_diameter_mean_um': ['2.017', '3.456', '1.789', '4.123', '2.888'],
        'gel_id': ['0057', '0058', '0059', '0060', '0061'],
        'numeric_float': [1.234, 5.678, 9.012, 3.456, 7.890],
        'numeric_int': [10, 20, 30, 40, 50],
        'numpy_array': np.array([100.5, 200.7, 300.9, 400.1, 500.3])
    }
    
    df = DataFrame(data, name="sample_data")
    
    print("✓ DataFrame created successfully!")
    print()
    
    print("IMPROVED OUTPUT:")
    print("-" * 40)
    print(df)
    print()
    
    print("KEY IMPROVEMENTS:")
    print("-" * 40)
    print("1. ✓ Numeric values (numeric_float, numeric_int, numpy_array) no longer have quotes")
    print("2. ✓ String values (volume_fraction, gel_id, etc.) still have quotes for clarity")
    print("3. ✓ Added 'Dtypes:' line showing detailed column data types")
    print("4. ✓ Float values formatted to 3 decimal places for consistency")
    print("5. ✓ JAX arrays display cleanly without Array() wrapper")
    print()
    
    print("STANDALONE DTYPES METHOD:")
    print("-" * 40)
    dtypes = df.dtypes()
    for col, dtype in dtypes.items():
        print(f"  {col:25} : {dtype}")
    print()
    
    # Test with JAX arrays if available
    try:
        import jax.numpy as jnp
        print("JAX ARRAYS DEMO:")
        print("-" * 40)
        jax_data = {
            'category': ['A', 'B', 'C'],
            'jax_int': jnp.array([1000, 2000, 3000]),
            'jax_float': jnp.array([123.456789, 234.567890, 345.678901])
        }
        jax_df = DataFrame(jax_data)
        print(jax_df)
        print()
        print("✓ JAX arrays are displayed cleanly without Array() wrapper")
        print()
    except ImportError:
        print("JAX not available - skipping JAX demo")
        print()
    
    print("SUMMARY:")
    print("-" * 40)
    print("The DataFrame printing now provides:")
    print("• Cleaner display without quotes around numeric values")
    print("• Detailed dtype information for each column")
    print("• Consistent float formatting (3 decimal places)")
    print("• Support for lists, numpy arrays, and JAX arrays")
    print("• Clear distinction between string and numeric data")
    print()
    print("This makes DataFrame output much more readable and informative!")

if __name__ == "__main__":
    main()
