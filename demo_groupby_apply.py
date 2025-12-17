"""
Demo: GroupBy.apply() - Applying transformations within groups

This demonstrates the powerful pattern: df.group_by('col').apply(func, 'value_col')

The key difference:
- group_by().agg(): Reduces each group to ONE row (e.g., mean, sum)
- group_by().apply(): Transforms values within each group, PRESERVES row count
"""

import jax.numpy as jnp
from jax import jit
from jaxframe import DataFrame


def demo_1_basic_transformation():
    """Demo 1: Basic transformation within groups."""
    print("\n" + "="*70)
    print("Demo 1: Basic Transformation Within Groups")
    print("="*70)
    
    df = DataFrame({
        'category': ['A', 'A', 'B', 'B', 'A'],
        'value': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    # Double values within each group
    result = df.group_by('category').apply(lambda x: x * 2, 'value')
    
    print("\nAfter doubling values within each group:")
    print(result)
    
    # Note: All rows preserved, transformation applied per-group


def demo_2_normalize_by_group():
    """Demo 2: Normalize values to [0, 1] within each group."""
    print("\n" + "="*70)
    print("Demo 2: Normalize to [0, 1] Within Each Group")
    print("="*70)
    
    df = DataFrame({
        'product': ['Electronics', 'Electronics', 'Electronics', 
                    'Clothing', 'Clothing', 'Clothing'],
        'price': jnp.array([100.0, 500.0, 1000.0, 20.0, 50.0, 100.0])
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    # Normalize prices to [0, 1] within each product category
    def normalize(x):
        return (x - jnp.min(x)) / (jnp.max(x) - jnp.min(x))
    
    result = df.group_by('product').apply(normalize, 'price', output_column='normalized')
    
    print("\nAfter normalization within product categories:")
    print(result)
    
    print("\nKey insight: Electronics prices scaled relative to $100-$1000 range")
    print("            Clothing prices scaled relative to $20-$100 range")


def demo_3_zscore_standardization():
    """Demo 3: Z-score standardization within groups."""
    print("\n" + "="*70)
    print("Demo 3: Z-Score Standardization by Department")
    print("="*70)
    
    df = DataFrame({
        'department': ['Sales', 'Sales', 'Sales', 'Engineering', 'Engineering', 'Engineering'],
        'salary': jnp.array([50000, 60000, 70000, 100000, 120000, 140000])
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    # Z-score: (x - mean) / std within each department
    def zscore(x):
        return (x - jnp.mean(x)) / jnp.std(x)
    
    result = df.group_by('department').apply(zscore, 'salary', output_column='zscore')
    
    print("\nAfter z-score standardization by department:")
    print(result)
    
    print("\nKey insight: Each department normalized to mean=0, std=1")
    print("            Allows fair comparison across departments with different scales")


def demo_4_ranking_within_groups():
    """Demo 4: Rank values within each group."""
    print("\n" + "="*70)
    print("Demo 4: Ranking Within Groups")
    print("="*70)
    
    df = DataFrame({
        'team': ['A', 'A', 'A', 'B', 'B', 'B'],
        'score': jnp.array([85, 92, 78, 95, 88, 90])
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    # Rank within each team
    def rank(x):
        return jnp.argsort(jnp.argsort(x)).astype(jnp.float32)
    
    result = df.group_by('team').apply(rank, 'score', output_column='rank')
    
    print("\nAfter ranking within teams:")
    print(result)
    
    print("\nKey insight: Rankings are computed WITHIN each team")
    print("            Team A: [85, 92, 78] -> ranks [1, 2, 0]")
    print("            Team B: [95, 88, 90] -> ranks [2, 0, 1]")


def demo_5_percentile_rank():
    """Demo 5: Compute percentile rank within groups."""
    print("\n" + "="*70)
    print("Demo 5: Percentile Rank by Department")
    print("="*70)
    
    df = DataFrame({
        'department': ['HR', 'HR', 'HR', 'IT', 'IT', 'IT'],
        'salary': jnp.array([45000, 50000, 55000, 80000, 90000, 100000])
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    def percentile_rank(x):
        """Compute percentile rank (0-100)."""
        ranks = jnp.argsort(jnp.argsort(x))
        return 100 * ranks / (len(x) - 1)
    
    result = df.group_by('department').apply(
        percentile_rank, 'salary', output_column='percentile'
    )
    
    print("\nAfter computing percentile ranks:")
    print(result)
    
    print("\nKey insight: Percentiles show position within department")
    print("            0th percentile = lowest in group")
    print("            100th percentile = highest in group")


def demo_6_deviation_from_group_mean():
    """Demo 6: Compute deviation from group mean."""
    print("\n" + "="*70)
    print("Demo 6: Deviation from Group Mean")
    print("="*70)
    
    df = DataFrame({
        'stock': ['AAPL', 'AAPL', 'AAPL', 'GOOG', 'GOOG', 'GOOG'],
        'price': jnp.array([150.0, 160.0, 155.0, 2800.0, 2900.0, 2850.0])
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    # Deviation from mean within each stock
    def deviation_from_mean(x):
        return x - jnp.mean(x)
    
    result = df.group_by('stock').apply(
        deviation_from_mean, 'price', output_column='deviation'
    )
    
    print("\nAfter computing deviations from group means:")
    print(result)
    
    print("\nKey insight: Shows how much each price deviates from its stock's average")
    print("            Positive = above average, Negative = below average")


def demo_7_multi_column_grouping():
    """Demo 7: Apply with multiple grouping columns."""
    print("\n" + "="*70)
    print("Demo 7: Multi-Column Grouping")
    print("="*70)
    
    df = DataFrame({
        'year': jnp.array([2020, 2020, 2021, 2021, 2020, 2021]),
        'quarter': jnp.array([1, 1, 1, 1, 2, 2]),
        'revenue': jnp.array([100.0, 150.0, 200.0, 250.0, 120.0, 280.0])
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    # Normalize within year-quarter groups
    def normalize(x):
        return (x - jnp.mean(x)) / (jnp.std(x) + 1e-8)
    
    result = df.group_by(['year', 'quarter']).apply(
        normalize, 'revenue', output_column='normalized'
    )
    
    print("\nAfter normalizing within year-quarter groups:")
    print(result)
    
    print("\nKey insight: Normalization applied to each year-quarter combination")


def demo_8_jax_functions():
    """Demo 8: Using JAX functions and JIT compilation."""
    print("\n" + "="*70)
    print("Demo 8: JAX Functions with JIT Compilation")
    print("="*70)
    
    df = DataFrame({
        'category': jnp.array([1, 1, 2, 2]),
        'value': jnp.array([4.0, 9.0, 16.0, 25.0])
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    # Use JAX's sqrt function
    result1 = df.group_by('category').apply(jnp.sqrt, 'value', output_column='sqrt')
    
    print("\nAfter applying jnp.sqrt:")
    print(result1)
    
    # JIT-compiled custom function
    @jit
    def scale_by_max(x):
        return x / jnp.max(x)
    
    result2 = df.group_by('category').apply(scale_by_max, 'value', output_column='scaled')
    
    print("\nAfter applying JIT-compiled scale_by_max:")
    print(result2)
    
    print("\nKey insight: Full JAX compatibility including JIT, vmap, grad")


def demo_9_chaining_apply_and_agg():
    """Demo 9: Chain apply() with agg() operations."""
    print("\n" + "="*70)
    print("Demo 9: Chaining Apply and Aggregate")
    print("="*70)
    
    df = DataFrame({
        'category': jnp.array([1, 1, 1, 2, 2, 2]),
        'value': jnp.array([10.0, 20.0, 30.0, 5.0, 15.0, 25.0])
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    # First: normalize within groups (apply)
    def zscore(x):
        return (x - jnp.mean(x)) / jnp.std(x)
    
    normalized = df.group_by('category').apply(zscore, 'value', output_column='zscore')
    
    print("\nAfter z-score normalization:")
    print(normalized)
    
    # Then: aggregate the normalized values (agg)
    result = normalized.group_by('category').agg({'zscore': ['mean', 'std']})
    
    print("\nAfter aggregating normalized values:")
    print(result)
    
    print("\nKey insight: apply() preserves rows, agg() reduces to one row per group")
    print("            Each group has mean ~0 and std ~1 after z-score normalization")


def demo_10_real_world_ml_preprocessing():
    """Demo 10: Real-world ML preprocessing by category."""
    print("\n" + "="*70)
    print("Demo 10: ML Feature Engineering by Category")
    print("="*70)
    
    df = DataFrame({
        'product_type': ['Electronics', 'Electronics', 'Electronics',
                        'Clothing', 'Clothing', 'Clothing'],
        'price': jnp.array([100.0, 500.0, 1000.0, 20.0, 50.0, 80.0]),
        'units_sold': jnp.array([50, 30, 10, 200, 150, 100])
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    # Log transform prices within each category
    def log_transform(x):
        return jnp.log(x + 1)
    
    result = df.group_by('product_type').apply(
        log_transform, 'price', output_column='log_price'
    )
    
    # Also normalize units_sold within categories
    def normalize(x):
        return (x - jnp.min(x)) / (jnp.max(x) - jnp.min(x) + 1e-8)
    
    result = result.group_by('product_type').apply(
        normalize, 'units_sold', output_column='norm_units'
    )
    
    print("\nAfter feature engineering:")
    print(result)
    
    print("\nKey insight: Different transformations for different product categories")
    print("            Log transform handles wide price ranges in Electronics")
    print("            Normalization scales units_sold relative to each category")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("JAXFrame: GroupBy.apply() Demonstrations")
    print("="*70)
    print("\nPattern: df.group_by('col').apply(func, 'value_col')")
    print("\nKey Features:")
    print("  • Apply transformations WITHIN each group")
    print("  • Preserves all rows (unlike agg() which reduces)")
    print("  • Full JAX compatibility (jit, vmap, grad)")
    print("  • Perfect for normalization, ranking, z-scores")
    
    demo_1_basic_transformation()
    demo_2_normalize_by_group()
    demo_3_zscore_standardization()
    demo_4_ranking_within_groups()
    demo_5_percentile_rank()
    demo_6_deviation_from_group_mean()
    demo_7_multi_column_grouping()
    demo_8_jax_functions()
    demo_9_chaining_apply_and_agg()
    demo_10_real_world_ml_preprocessing()
    
    print("\n" + "="*70)
    print("Summary: When to use what?")
    print("="*70)
    print("\n• DataFrame.apply(): Transform columns across entire DataFrame")
    print("• GroupBy.agg():     Reduce each group to ONE summary row")
    print("• GroupBy.apply():   Transform values WITHIN groups, keep all rows")
    print("\nExamples:")
    print("  df.apply(jnp.log, 'price')                 # Log all prices")
    print("  df.group_by('cat').agg({'x': 'mean'})      # Mean per category")
    print("  df.group_by('cat').apply(zscore, 'x')      # Z-score within each category")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
