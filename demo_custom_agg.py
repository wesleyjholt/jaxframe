"""
Demo: Custom Aggregation Functions with GroupBy.agg()

This demonstrates how to use custom JAX-compatible functions
alongside built-in aggregations in group_by().agg().
"""

import jax.numpy as jnp
from jax import jit
from jaxframe import DataFrame


def demo_1_simple_custom_function():
    """Demo 1: Using a simple custom aggregation function."""
    print("\n" + "="*70)
    print("Demo 1: Simple Custom Aggregation - Range (Max - Min)")
    print("="*70)
    
    df = DataFrame({
        'category': ['A', 'A', 'A', 'B', 'B', 'B'],
        'value': jnp.array([10, 15, 20, 100, 150, 200])
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    # Custom function: compute range (max - min)
    def range_func(x):
        return jnp.max(x) - jnp.min(x)
    
    result = df.group_by('category').agg({'value': range_func})
    
    print("\nRange within each category:")
    print(result)
    print("\nCategory A: 20 - 10 = 10")
    print("Category B: 200 - 100 = 100")


def demo_2_jax_built_in_functions():
    """Demo 2: Using JAX built-in functions."""
    print("\n" + "="*70)
    print("Demo 2: JAX Built-in Functions (Median, Percentiles)")
    print("="*70)
    
    df = DataFrame({
        'department': ['Sales', 'Sales', 'Sales', 'Engineering', 'Engineering', 'Engineering'],
        'salary': jnp.array([50000, 60000, 70000, 90000, 100000, 110000])
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    # Use JAX's median and variance functions
    result = df.group_by('department').agg({
        'salary': [jnp.median, jnp.var]
    })
    
    print("\nMedian and variance by department:")
    print(result)


def demo_3_mix_builtin_and_custom():
    """Demo 3: Mixing built-in and custom aggregations."""
    print("\n" + "="*70)
    print("Demo 3: Mix Built-in and Custom Aggregations")
    print("="*70)
    
    df = DataFrame({
        'product': ['A', 'A', 'A', 'B', 'B', 'B'],
        'price': jnp.array([10.0, 12.0, 14.0, 100.0, 120.0, 140.0])
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    # Coefficient of variation: std / mean
    def cv(x):
        return jnp.std(x) / jnp.mean(x)
    
    result = df.group_by('product').agg({
        'price': ['mean', 'std', cv]
    })
    
    print("\nMean, std, and coefficient of variation by product:")
    print(result)
    print("\nCV shows relative variability (lower = more consistent)")


def demo_4_percentiles():
    """Demo 4: Computing multiple percentiles."""
    print("\n" + "="*70)
    print("Demo 4: Multiple Percentiles (25th, 50th, 75th)")
    print("="*70)
    
    df = DataFrame({
        'region': ['North', 'North', 'North', 'North', 
                   'South', 'South', 'South', 'South'],
        'sales': jnp.array([100, 150, 200, 250, 50, 75, 100, 125])
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    # Define percentile functions
    def p25(x):
        return jnp.percentile(x, 25)
    
    def p75(x):
        return jnp.percentile(x, 75)
    
    result = df.group_by('region').agg({
        'sales': [p25, jnp.median, p75]
    })
    
    print("\nQuartile analysis by region:")
    print(result)


def demo_5_geometric_and_harmonic_mean():
    """Demo 5: Geometric and harmonic means."""
    print("\n" + "="*70)
    print("Demo 5: Geometric and Harmonic Means")
    print("="*70)
    
    df = DataFrame({
        'category': jnp.array([1, 1, 1, 2, 2, 2]),
        'rate': jnp.array([1.05, 1.10, 1.15, 1.02, 1.03, 1.04])
    })
    
    print("\nOriginal DataFrame (growth rates):")
    print(df)
    
    def geometric_mean(x):
        """Geometric mean: exp(mean(log(x)))."""
        return jnp.exp(jnp.mean(jnp.log(x)))
    
    def harmonic_mean(x):
        """Harmonic mean: n / sum(1/x)."""
        return len(x) / jnp.sum(1.0 / x)
    
    result = df.group_by('category').agg({
        'rate': ['mean', geometric_mean, harmonic_mean]
    })
    
    print("\nDifferent types of means:")
    print(result)
    print("\nGeometric mean: better for multiplicative processes (e.g., growth rates)")
    print("Harmonic mean: better for rates and ratios")


def demo_6_iqr_for_outlier_detection():
    """Demo 6: Interquartile range for outlier detection."""
    print("\n" + "="*70)
    print("Demo 6: Interquartile Range (IQR) for Outlier Detection")
    print("="*70)
    
    df = DataFrame({
        'sensor': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'reading': jnp.array([10, 11, 12, 50, 100, 101, 102, 200])
    })
    
    print("\nOriginal DataFrame (sensor readings with outliers):")
    print(df)
    
    def iqr(x):
        """Interquartile range: Q3 - Q1."""
        return jnp.percentile(x, 75) - jnp.percentile(x, 25)
    
    result = df.group_by('sensor').agg({
        'reading': [jnp.median, iqr]
    })
    
    print("\nMedian and IQR by sensor:")
    print(result)
    print("\nIQR measures spread of middle 50% of data")
    print("Outliers typically defined as values beyond Q1-1.5*IQR or Q3+1.5*IQR")


def demo_7_jit_compiled_functions():
    """Demo 7: Using JIT-compiled custom functions."""
    print("\n" + "="*70)
    print("Demo 7: JIT-Compiled Custom Functions")
    print("="*70)
    
    df = DataFrame({
        'group': jnp.array([1, 1, 1, 2, 2, 2]),
        'value': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    @jit
    def root_mean_square(x):
        """RMS: sqrt(mean(x^2))."""
        return jnp.sqrt(jnp.mean(x ** 2))
    
    result = df.group_by('group').agg({
        'value': ['mean', root_mean_square]
    })
    
    print("\nMean and RMS by group:")
    print(result)
    print("\nJIT compilation makes repeated calls faster!")


def demo_8_lambda_functions():
    """Demo 8: Using lambda functions for quick aggregations."""
    print("\n" + "="*70)
    print("Demo 8: Lambda Functions for Quick Aggregations")
    print("="*70)
    
    df = DataFrame({
        'category': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
        'value': jnp.array([10, 20, 30, 5, 15, 25])
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    result = df.group_by('category').agg({
        'value': [
            lambda x: jnp.sum(x),           # Total
            lambda x: jnp.max(x) / jnp.min(x),  # Max/min ratio
            lambda x: jnp.sum(x ** 2)       # Sum of squares
        ]
    })
    
    print("\nCustom aggregations using lambdas:")
    print(result)


def demo_9_multi_column_custom_agg():
    """Demo 9: Custom aggregations on multiple columns."""
    print("\n" + "="*70)
    print("Demo 9: Custom Aggregations on Multiple Columns")
    print("="*70)
    
    df = DataFrame({
        'team': ['A', 'A', 'A', 'B', 'B', 'B'],
        'points': jnp.array([10, 15, 20, 8, 12, 16]),
        'assists': jnp.array([5, 8, 12, 3, 6, 9])
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    def efficiency_rating(x):
        """Simple efficiency: mean * consistency (1 - cv)."""
        cv = jnp.std(x) / jnp.mean(x)
        return jnp.mean(x) * (1 - cv)
    
    result = df.group_by('team').agg({
        'points': ['mean', efficiency_rating],
        'assists': ['mean', efficiency_rating]
    })
    
    print("\nEfficiency rating by team:")
    print(result)
    print("\nHigher efficiency = high mean + low variability")


def demo_10_advanced_statistical_measures():
    """Demo 10: Advanced statistical measures."""
    print("\n" + "="*70)
    print("Demo 10: Advanced Statistical Measures")
    print("="*70)
    
    df = DataFrame({
        'treatment': ['Control', 'Control', 'Control', 'Treatment', 'Treatment', 'Treatment'],
        'response': jnp.array([5.2, 5.5, 5.3, 7.1, 7.5, 7.3])
    })
    
    print("\nOriginal DataFrame (experiment results):")
    print(df)
    
    def skewness(x):
        """Simplified skewness measure."""
        mean = jnp.mean(x)
        std = jnp.std(x)
        return jnp.mean(((x - mean) / std) ** 3)
    
    def sem(x):
        """Standard error of mean."""
        return jnp.std(x) / jnp.sqrt(len(x))
    
    result = df.group_by('treatment').agg({
        'response': ['mean', 'std', sem, skewness]
    })
    
    print("\nStatistical summary by treatment:")
    print(result)
    print("\nSEM: Standard error of the mean (uncertainty in mean estimate)")
    print("Skewness: Measure of asymmetry in distribution")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("JAXFrame: Custom Aggregation Functions Demo")
    print("="*70)
    print("\nFeatures:")
    print("  • Use any JAX-compatible function as aggregation")
    print("  • Mix built-in ('sum', 'mean', 'std', etc.) with custom")
    print("  • Support for lambda functions, named functions, JIT-compiled")
    print("  • Perfect for advanced statistics and domain-specific metrics")
    
    demo_1_simple_custom_function()
    demo_2_jax_built_in_functions()
    demo_3_mix_builtin_and_custom()
    demo_4_percentiles()
    demo_5_geometric_and_harmonic_mean()
    demo_6_iqr_for_outlier_detection()
    demo_7_jit_compiled_functions()
    demo_8_lambda_functions()
    demo_9_multi_column_custom_agg()
    demo_10_advanced_statistical_measures()
    
    print("\n" + "="*70)
    print("Summary: Custom Aggregation Capabilities")
    print("="*70)
    print("\n✓ Built-in: 'sum', 'mean', 'std', 'min', 'max', 'count'")
    print("✓ JAX functions: jnp.median, jnp.var, jnp.percentile, etc.")
    print("✓ Custom functions: Any JAX-compatible scalar-returning function")
    print("✓ Lambda functions: Quick inline aggregations")
    print("✓ JIT-compiled: Performance optimization for repeated use")
    print("✓ Mix and match: Combine any of the above in one call")
    print("\nExamples:")
    print("  df.group_by('cat').agg({'x': 'mean'})")
    print("  df.group_by('cat').agg({'x': jnp.median})")
    print("  df.group_by('cat').agg({'x': lambda x: jnp.percentile(x, 90)})")
    print("  df.group_by('cat').agg({'x': ['mean', jnp.median, custom_func]})")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
