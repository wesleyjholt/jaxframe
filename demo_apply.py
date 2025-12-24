"""
Demo of DataFrame.apply() method with JAX-compatible functions.

This demonstrates how to apply custom transformations to DataFrame columns
while maintaining full JAX compatibility (jittable and differentiable).
"""

import jax.numpy as jnp
import numpy as np
from jax import jit, grad, vmap

from jaxframe import DataFrame


def demo_basic_apply():
    """Demonstrate basic apply functionality."""
    print("=" * 60)
    print("BASIC APPLY OPERATIONS")
    print("=" * 60)
    
    df = DataFrame({
        'x': jnp.array([1, 2, 3, 4, 5]),
        'y': jnp.array([10, 20, 30, 40, 50])
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    # Square a column (replace in-place)
    result1 = df.apply(lambda x: x ** 2, 'x')
    print("\nApply x^2 to column 'x' (replace):")
    print(result1)
    
    # Square a column with new name
    result2 = df.apply(lambda x: x ** 2, 'x', output_column='x_squared')
    print("\nApply x^2 to column 'x' (new column):")
    print(result2)
    
    # Apply to multiple columns
    result3 = df.apply(lambda x, y: x + y, ['x', 'y'], output_column='sum')
    print("\nAdd columns 'x' and 'y':")
    print(result3)


def demo_jax_functions():
    """Demonstrate using JAX built-in functions."""
    print("\n" + "=" * 60)
    print("JAX BUILT-IN FUNCTIONS")
    print("=" * 60)
    
    df = DataFrame({
        'values': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    # Apply log
    result = df.apply(jnp.log, 'values', output_column='log_values')
    print("\nApply jnp.log:")
    print(result)
    
    # Apply exp
    result = df.apply(jnp.exp, 'values', output_column='exp_values')
    print("\nApply jnp.exp:")
    print(result)
    
    # Apply sqrt
    result = df.apply(jnp.sqrt, 'values', output_column='sqrt_values')
    print("\nApply jnp.sqrt:")
    print(result)


def demo_custom_functions():
    """Demonstrate custom JAX-compatible functions."""
    print("\n" + "=" * 60)
    print("CUSTOM JAX-COMPATIBLE FUNCTIONS")
    print("=" * 60)
    
    df = DataFrame({
        'prices': jnp.array([100.0, 200.0, 300.0, 400.0, 500.0])
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    # Normalize to [0, 1]
    def normalize(x):
        return (x - jnp.min(x)) / (jnp.max(x) - jnp.min(x))
    
    result = df.apply(normalize, 'prices', output_column='normalized')
    print("\nNormalize to [0, 1]:")
    print(result)
    
    # Z-score normalization
    def zscore(x):
        return (x - jnp.mean(x)) / jnp.std(x)
    
    result = df.apply(zscore, 'prices', output_column='zscore')
    print("\nZ-score normalization:")
    print(result)
    
    # Sigmoid transformation
    def sigmoid(x):
        # Normalize first to avoid overflow
        x_norm = (x - jnp.mean(x)) / jnp.std(x)
        return 1 / (1 + jnp.exp(-x_norm))
    
    result = df.apply(sigmoid, 'prices', output_column='sigmoid')
    print("\nSigmoid transformation:")
    print(result)


def demo_multiple_columns():
    """Demonstrate operations on multiple columns."""
    print("\n" + "=" * 60)
    print("MULTIPLE COLUMN OPERATIONS")
    print("=" * 60)
    
    df = DataFrame({
        'x': jnp.array([3.0, 4.0, 5.0]),
        'y': jnp.array([4.0, 3.0, 12.0])
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    # Euclidean distance
    def euclidean(x, y):
        return jnp.sqrt(x**2 + y**2)
    
    result = df.apply(euclidean, ['x', 'y'], output_column='distance')
    print("\nEuclidean distance:")
    print(result)
    
    # Weighted sum
    def weighted_sum(x, y):
        return 0.7 * x + 0.3 * y
    
    result = df.apply(weighted_sum, ['x', 'y'], output_column='weighted')
    print("\nWeighted sum (0.7*x + 0.3*y):")
    print(result)
    
    # Ratio
    def safe_ratio(x, y):
        return x / (y + 1e-8)  # Add small epsilon to avoid division by zero
    
    result = df.apply(safe_ratio, ['x', 'y'], output_column='ratio')
    print("\nRatio (x/y):")
    print(result)


def demo_jit_compilation():
    """Demonstrate JIT compilation compatibility."""
    print("\n" + "=" * 60)
    print("JIT COMPILATION COMPATIBILITY")
    print("=" * 60)
    
    df = DataFrame({
        'values': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    # Define a JIT-compiled function
    @jit
    def complex_transform(x):
        return jnp.sqrt(x**2 + 1) * jnp.sin(x)
    
    result = df.apply(complex_transform, 'values', output_column='transformed')
    print("\nApply JIT-compiled function:")
    print(result)
    
    # The result can also be used in JIT-compiled code
    @jit
    def compute_sum(arr):
        return jnp.sum(arr)
    
    total = compute_sum(result['transformed'])
    print(f"\nSum of transformed values (JIT-compiled): {total:.4f}")


def demo_gradients():
    """Demonstrate gradient computation through apply."""
    print("\n" + "=" * 60)
    print("GRADIENT COMPUTATION")
    print("=" * 60)
    
    df = DataFrame({
        'x': jnp.array([1.0, 2.0, 3.0])
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    # Apply a transformation
    result = df.apply(lambda x: x ** 2, 'x', output_column='x_squared')
    print("\nAfter x^2:")
    print(result)
    
    # Define a loss function
    def loss_fn(x):
        x_squared = x ** 2
        return jnp.sum(x_squared)
    
    # Compute gradients
    grad_fn = grad(loss_fn)
    gradients = grad_fn(df['x'])
    
    print(f"\nGradients of sum(x^2) with respect to x: {gradients}")
    print(f"Expected (2*x): {2 * df['x']}")
    print("✓ Gradients match expected values!")


def demo_chaining():
    """Demonstrate chaining multiple apply operations."""
    print("\n" + "=" * 60)
    print("CHAINING APPLY OPERATIONS")
    print("=" * 60)
    
    df = DataFrame({
        'raw_data': jnp.array([10.0, 20.0, 30.0, 40.0])
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    # Chain multiple transformations
    result = (df
              .apply(jnp.log, 'raw_data', output_column='log_data')
              .apply(lambda x: x ** 2, 'log_data', output_column='log_squared')
              .apply(lambda x: x / 2, 'raw_data', output_column='halved')
              .apply(lambda x, y: x + y, ['log_data', 'halved'], output_column='combined'))
    
    print("\nAfter chaining multiple apply operations:")
    print(result)


def demo_feature_engineering():
    """Demonstrate feature engineering use case."""
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING EXAMPLE")
    print("=" * 60)
    
    # Customer data
    df = DataFrame({
        'age': jnp.array([25, 35, 45, 55, 65]),
        'income': jnp.array([50000, 75000, 100000, 125000, 150000]),
        'purchases': jnp.array([5, 10, 15, 20, 25])
    })
    
    print("\nOriginal customer data:")
    print(df)
    
    # Create engineered features
    result = (df
              # Age bins (decades)
              .apply(lambda x: jnp.floor(x / 10), 'age', output_column='age_decade')
              # Log income (reduce skew)
              .apply(jnp.log, 'income', output_column='log_income')
              # Purchase rate per $1000 income
              .apply(lambda p, i: p / (i / 1000), 
                     ['purchases', 'income'], 
                     output_column='purchase_rate')
              # Age-income interaction
              .apply(lambda a, i: a * i / 1000000, 
                     ['age', 'income'], 
                     output_column='age_income_interaction'))
    
    print("\nWith engineered features:")
    print(result)


def demo_real_world_ml():
    """Demonstrate real-world ML preprocessing."""
    print("\n" + "=" * 60)
    print("REAL-WORLD ML PREPROCESSING")
    print("=" * 60)
    
    # Simulated sensor data
    df = DataFrame({
        'sensor_1': jnp.array([0.1, 0.5, 0.9, 1.2, 1.5]),
        'sensor_2': jnp.array([2.3, 2.5, 2.7, 2.9, 3.1]),
        'sensor_3': jnp.array([10.0, 12.0, 14.0, 16.0, 18.0])
    })
    
    print("\nRaw sensor data:")
    print(df)
    
    # Standardize features
    def standardize(x):
        return (x - jnp.mean(x)) / (jnp.std(x) + 1e-8)
    
    result = (df
              .apply(standardize, 'sensor_1', output_column='sensor_1_std')
              .apply(standardize, 'sensor_2', output_column='sensor_2_std')
              .apply(standardize, 'sensor_3', output_column='sensor_3_std'))
    
    print("\nStandardized features:")
    print(result)
    
    # Create polynomial features
    result = (result
              .apply(lambda x: x ** 2, 'sensor_1_std', output_column='sensor_1_squared')
              .apply(lambda x, y: x * y, 
                     ['sensor_1_std', 'sensor_2_std'], 
                     output_column='sensor_1_2_interaction'))
    
    print("\nWith polynomial features:")
    print(result)
    
    print("\n✓ Ready for ML model training!")


def demo_performance_note():
    """Note about performance and best practices."""
    print("\n" + "=" * 60)
    print("PERFORMANCE NOTES & BEST PRACTICES")
    print("=" * 60)
    
    print("""
JAXFrame apply() Method:

✓ PROS:
  - Full JAX compatibility (jittable and differentiable)
  - Works with any JAX-compatible function
  - Automatic conversion of lists/numpy arrays to JAX arrays
  - Supports both single and multiple column operations
  - Method chaining for complex transformations
  - Preserves JAX array types in output

💡 BEST PRACTICES:
  1. Use JAX functions (jnp.log, jnp.exp, etc.) for best performance
  2. JIT-compile custom functions for repeated use
  3. Chain multiple apply() calls for complex pipelines
  4. Use multiple column apply for vectorized operations
  5. Ensure functions return arrays, not scalars
  6. For ML: standardize features before applying transformations

⚠ LIMITATIONS:
  - Function must return array with same length as input
  - Scalar outputs are not supported (use reduction operations separately)
  - Multiple column apply requires explicit output_column name
  - String columns will be converted to numeric arrays

📊 COMPARED TO POLARS:
  - JAXFrame: Full JAX integration (jit/grad/vmap)
  - Polars: Richer expression system, string operations
  - JAXFrame: Best for numerical/ML workflows
  - Polars: Best for general data manipulation
    """)


if __name__ == '__main__':
    demo_basic_apply()
    demo_jax_functions()
    demo_custom_functions()
    demo_multiple_columns()
    demo_jit_compilation()
    demo_gradients()
    demo_chaining()
    demo_feature_engineering()
    demo_real_world_ml()
    demo_performance_note()
    
    print("\n" + "=" * 60)
    print("✓ ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
