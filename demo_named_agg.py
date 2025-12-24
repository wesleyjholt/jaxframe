"""
Demo: Named Custom Aggregations using Tuple Syntax

This demo shows how to use the (name, function) tuple syntax to explicitly
name custom aggregation functions for better readability and control over
result column names.
"""

import jax.numpy as jnp
from jaxframe import DataFrame


print("=" * 70)
print("Demo 1: Basic Named Custom Function")
print("=" * 70)
print("Using tuple to provide an explicit name for a custom function.")
print()

df1 = DataFrame({
    'category': ['A', 'A', 'B', 'B'],
    'value': jnp.array([1.0, 5.0, 10.0, 20.0])
})

print("Original DataFrame:")
print(df1)
print()

# Use tuple (name, function) to explicitly name the aggregation
result1 = df1.group_by('category').agg({
    'value': [
        'mean',
        ('range', lambda x: jnp.max(x) - jnp.min(x))
    ]
})

print("Result with named 'range' function:")
print(result1)
print("Column names:", result1.columns)
print()


print("=" * 70)
print("Demo 2: Multiple Named Functions")
print("=" * 70)
print("Using multiple named functions for percentile calculations.")
print()

df2 = DataFrame({
    'group': jnp.array([1, 1, 1, 2, 2, 2]),
    'score': jnp.array([85, 90, 95, 70, 75, 80])
})

print("Original DataFrame:")
print(df2)
print()

result2 = df2.group_by('group').agg({
    'score': [
        ('p25', lambda x: jnp.percentile(x, 25)),
        ('median', jnp.median),
        ('p75', lambda x: jnp.percentile(x, 75)),
        ('iqr', lambda x: jnp.percentile(x, 75) - jnp.percentile(x, 25))
    ]
})

print("Result with multiple named percentile functions:")
print(result2)
print()


print("=" * 70)
print("Demo 3: Financial Metrics with Descriptive Names")
print("=" * 70)
print("Computing stock price metrics with clear, descriptive names.")
print()

df3 = DataFrame({
    'stock': ['AAPL', 'AAPL', 'AAPL', 'GOOG', 'GOOG', 'GOOG'],
    'price': jnp.array([150.0, 155.0, 160.0, 2800.0, 2850.0, 2900.0])
})

print("Original DataFrame:")
print(df3)
print()

result3 = df3.group_by('stock').agg({
    'price': [
        ('low', jnp.min),
        ('high', jnp.max),
        ('avg', jnp.mean),
        ('volatility', jnp.std),
        ('range', lambda x: jnp.max(x) - jnp.min(x))
    ]
})

print("Result with financial metrics:")
print(result3)
print()


print("=" * 70)
print("Demo 4: Mixing Built-in, Unnamed, and Named Functions")
print("=" * 70)
print("Demonstrating all three types of aggregations together.")
print()

df4 = DataFrame({
    'category': jnp.array([1, 1, 1, 2, 2, 2]),
    'value': jnp.array([10, 15, 20, 30, 35, 40])
})

print("Original DataFrame:")
print(df4)
print()

result4 = df4.group_by('category').agg({
    'value': [
        'sum',                                           # Built-in string
        'mean',                                          # Built-in string
        jnp.median,                                      # Unnamed function (auto-named)
        jnp.std,                                         # Unnamed function (auto-named)
        ('p90', lambda x: jnp.percentile(x, 90)),       # Named tuple
        ('spread', lambda x: jnp.max(x) - jnp.min(x))  # Named tuple
    ]
})

print("Result mixing all three types:")
print(result4)
print("Column names:", result4.columns)
print()


print("=" * 70)
print("Demo 5: Sensor Quality Metrics")
print("=" * 70)
print("Computing quality metrics for sensor readings.")
print()

df5 = DataFrame({
    'sensor_id': ['S1', 'S1', 'S1', 'S2', 'S2', 'S2'],
    'reading': jnp.array([10.0, 10.1, 9.9, 50.0, 51.0, 49.0])
})

print("Original DataFrame:")
print(df5)
print()

def coefficient_of_variation(x):
    """CV = std/mean, a normalized measure of dispersion."""
    return jnp.std(x) / jnp.mean(x)

result5 = df5.group_by('sensor_id').agg({
    'reading': [
        ('average', jnp.mean),
        ('precision', jnp.std),
        ('cv', coefficient_of_variation),
        ('min_reading', jnp.min),
        ('max_reading', jnp.max)
    ]
})

print("Result with sensor quality metrics:")
print(result5)
print()


print("=" * 70)
print("Demo 6: Survey Analysis")
print("=" * 70)
print("Analyzing survey responses with descriptive statistics.")
print()

df6 = DataFrame({
    'question': ['Q1', 'Q1', 'Q1', 'Q1', 'Q2', 'Q2', 'Q2', 'Q2'],
    'response': jnp.array([1, 2, 3, 5, 2, 3, 3, 4])
})

print("Original DataFrame:")
print(df6)
print()

result6 = df6.group_by('question').agg({
    'response': [
        ('avg_score', jnp.mean),
        ('median_score', jnp.median),
        ('std_dev', jnp.std),
        ('min_score', jnp.min),
        ('max_score', jnp.max)
    ]
})

print("Result with survey statistics:")
print(result6)
print()


print("=" * 70)
print("Demo 7: Multiple Columns with Same Named Functions")
print("=" * 70)
print("Applying the same named functions to multiple columns.")
print()

df7 = DataFrame({
    'team': ['A', 'A', 'B', 'B'],
    'points': jnp.array([10, 20, 30, 40]),
    'assists': jnp.array([5, 8, 12, 15])
})

print("Original DataFrame:")
print(df7)
print()

result7 = df7.group_by('team').agg({
    'points': [
        ('avg', jnp.mean),
        ('max', jnp.max)
    ],
    'assists': [
        ('avg', jnp.mean),
        ('max', jnp.max)
    ]
})

print("Result with consistent naming across columns:")
print(result7)
print("Note: Same function names ('avg', 'max') used for both columns")
print()


print("=" * 70)
print("Demo 8: Complex Custom Metrics")
print("=" * 70)
print("Computing complex, domain-specific metrics with clear names.")
print()

df8 = DataFrame({
    'experiment': ['A', 'A', 'A', 'B', 'B', 'B'],
    'measurement': jnp.array([98.5, 101.2, 99.8, 195.0, 203.5, 198.7])
})

print("Original DataFrame:")
print(df8)
print()

def relative_std_dev(x):
    """Relative standard deviation (RSD) as percentage."""
    return (jnp.std(x) / jnp.mean(x)) * 100

def signal_to_noise(x):
    """Signal-to-noise ratio."""
    return jnp.mean(x) / jnp.std(x)

result8 = df8.group_by('experiment').agg({
    'measurement': [
        ('mean', jnp.mean),
        ('std', jnp.std),
        ('rsd_pct', relative_std_dev),
        ('snr', signal_to_noise),
        ('cv', lambda x: jnp.std(x) / jnp.mean(x))
    ]
})

print("Result with complex quality metrics:")
print(result8)
print()


print("=" * 70)
print("Demo 9: Single Named Function (No Suffix)")
print("=" * 70)
print("When using a single named function, no suffix is added to the column.")
print()

df9 = DataFrame({
    'group': jnp.array([1, 1, 2, 2]),
    'value': jnp.array([1.0, 2.0, 3.0, 4.0])
})

print("Original DataFrame:")
print(df9)
print()

result9 = df9.group_by('group').agg({
    'value': ('custom_metric', lambda x: jnp.sum(x ** 2))
})

print("Result with single named function:")
print(result9)
print("Column names:", result9.columns)
print("Note: Column is named 'value', not 'value_custom_metric'")
print()


print("=" * 70)
print("Demo 10: Comparing Named vs Unnamed Functions")
print("=" * 70)
print("Side-by-side comparison of naming strategies.")
print()

df10 = DataFrame({
    'category': ['A', 'A', 'B', 'B'],
    'value': jnp.array([10, 20, 30, 40])
})

print("Original DataFrame:")
print(df10)
print()

# Unnamed function (auto-named from __name__)
result_unnamed = df10.group_by('category').agg({
    'value': [jnp.mean, jnp.median, jnp.std]
})

print("Result with UNNAMED functions (auto-naming):")
print(result_unnamed)
print("Column names:", result_unnamed.columns)
print()

# Named functions (explicit control)
result_named = df10.group_by('category').agg({
    'value': [
        ('average', jnp.mean),
        ('middle', jnp.median),
        ('spread', jnp.std)
    ]
})

print("Result with NAMED functions (explicit control):")
print(result_named)
print("Column names:", result_named.columns)
print()
print("Benefit: Named tuples give you complete control over result column names!")
print()


print("=" * 70)
print("Summary: When to Use Named Tuples")
print("=" * 70)
print("""
Named tuple syntax: ('name', function)

✅ USE WHEN:
  - You want clear, readable column names
  - The function's __name__ is unclear (e.g., lambda, '<lambda>')
  - You want consistent naming across multiple columns
  - You're computing domain-specific metrics (e.g., 'p90', 'cv', 'snr')
  - You want shorter names than the function's __name__

❌ DON'T NEED WHEN:
  - Using built-in strings ('mean', 'sum', etc.)
  - Function has a good __name__ (e.g., jnp.median → 'median')
  - Auto-naming is sufficient

SYNTAX EXAMPLES:
  ('p90', lambda x: jnp.percentile(x, 90))    # Lambda with clear name
  ('range', lambda x: jnp.max(x) - jnp.min(x)) # Complex calculation
  ('cv', coefficient_of_variation)             # Descriptive short name
  ('avg', jnp.mean)                            # Preferred name vs 'mean'
""")
