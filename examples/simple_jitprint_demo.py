"""
Simple demonstration of JIT-compatible printing for JAXFrame objects.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jaxframe import DataFrame, MaskedArray, jit_print_dataframe, jit_print_masked_array


def main():
    print("JAXFrame JIT Print Demo")
    print("=" * 40)
    
    # Create test data
    df = DataFrame({
        'id': [1, 2, 3],
        'value': [10.5, 20.7, 30.9],
        'name': ['apple', 'banana', 'cherry']
    })
    
    data = jnp.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]])
    mask = np.array([[True, False], [True, True], [False, True]])
    index_df = DataFrame({'row': [1, 2, 3], 'type': ['A', 'B', 'A']})
    ma = MaskedArray(data, mask, index_df)
    
    print("\n1. Regular printing (outside JIT):")
    print("DataFrame:")
    print(df)
    print("\nMaskedArray:")
    print(ma)
    
    print("\n2. JIT printing (inside JIT-compiled function):")
    
    # DataFrame in JIT
    def process_df(x, dataframe):
        jax.debug.print("DataFrame inside JIT:")
        jit_print_dataframe(dataframe)
        return x * 2
    
    process_df_jit = jax.jit(process_df, static_argnames=['dataframe'])
    
    # MaskedArray in JIT  
    def process_ma(x, masked_array):
        jax.debug.print("MaskedArray inside JIT:")
        jit_print_masked_array(masked_array)
        return x + 1
    
    process_ma_jit = jax.jit(process_ma, static_argnames=['masked_array'])
    
    # Run the JIT functions
    print("Running JIT functions...")
    result1 = process_df_jit(jnp.array([1.0, 2.0]), df)
    result2 = process_ma_jit(jnp.array([3.0, 4.0]), ma)
    
    print(f"\nResults: DataFrame: {result1}, MaskedArray: {result2}")
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
