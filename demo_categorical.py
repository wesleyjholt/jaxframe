"""
Demo showcasing the categorical tracking and improved dtypes features.
"""
import jax.numpy as jnp
import numpy as np
from src.jaxframe import DataFrame

print("="*80)
print("JAXFRAME: CATEGORICAL TRACKING & IMPROVED DTYPES DEMO")
print("="*80)

# Create a DataFrame with mixed types
df = DataFrame({
    'customer_id': [1001, 1002, 1003, 1004],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'age': [25, 32, 28, 45],
    'department': ['Sales', 'Engineering', 'Sales', 'Marketing'],
    'salary': [55000.0, 75000.0, 52000.0, 68000.0],
    'years_employed': [2, 5, 1, 8],
    'is_manager': [False, True, False, True],
    'performance_score': jnp.array([4.2, 4.8, 3.9, 4.5])
})

print("\n📊 DataFrame:")
print(df)

print("\n" + "="*80)
print("1. DTYPES - Element Types (Not Container Types)")
print("="*80)
print("\nBefore fix: dtypes would return 'list[int]', 'list[str]', etc.")
print("After fix: dtypes returns actual element types\n")
print(f"Dtypes: {df.dtypes}")
print("\n✓ Notice: 'str', 'int', 'float' (not 'list[str]', 'list[int]', etc.)")

print("\n" + "="*80)
print("2. AUTOMATIC CATEGORICAL INFERENCE")
print("="*80)
print("\nDefault categorical status:")
print(f"{df.categorical}\n")

print("Rules applied:")
print("  • customer_id (int):        Categorical ✓ (discrete ID)")
print("  • name (str):               Categorical ✓ (always categorical)")
print("  • age (int):                Categorical ✓ (default for ints)")
print("  • department (str):         Categorical ✓ (always categorical)")
print("  • salary (float):           Non-categorical ✗ (always continuous)")
print("  • years_employed (int):     Categorical ✓ (default for ints)")
print("  • is_manager (bool):        Categorical ✓ (default for bools)")
print("  • performance_score (float): Non-categorical ✗ (always continuous)")

print("\n" + "="*80)
print("3. QUERY CATEGORICAL STATUS")
print("="*80)

cat_cols = df.get_categorical_columns()
non_cat_cols = df.get_non_categorical_columns()

print(f"\nCategorical columns:     {cat_cols}")
print(f"Non-categorical columns: {non_cat_cols}")

print("\n" + "="*80)
print("4. MODIFY CATEGORICAL STATUS")
print("="*80)

print("\n4a. Make 'age' and 'years_employed' non-categorical:")
print("    (These are better treated as continuous variables)")

df2 = df.as_non_categorical(['age', 'years_employed'])
print(f"\nNew categorical status:")
for col in ['age', 'years_employed', 'salary', 'performance_score']:
    status = "Categorical" if df2.is_categorical(col) else "Non-categorical"
    print(f"  • {col:20s} → {status}")

print("\n4b. Keep 'customer_id' categorical:")
print("    (IDs should remain categorical for proper grouping)")
print(f"\n  • customer_id is_categorical: {df2.is_categorical('customer_id')}")

print("\n" + "="*80)
print("5. VALIDATION: FLOAT CANNOT BE CATEGORICAL")
print("="*80)

try:
    df.as_categorical('salary')
    print("❌ Should have raised error!")
except ValueError as e:
    print(f"\n✓ Correctly rejected: {e}")

print("\n" + "="*80)
print("6. VALIDATION: STRING CANNOT BE NON-CATEGORICAL")
print("="*80)

try:
    df.as_non_categorical('name')
    print("❌ Should have raised error!")
except ValueError as e:
    print(f"\n✓ Correctly rejected: {e}")

print("\n" + "="*80)
print("7. EXPLICIT CONTROL IN CONSTRUCTOR")
print("="*80)

print("\nCreate DataFrame with explicit categorical specification:")

df3 = DataFrame(
    {
        'employee_id': [1, 2, 3],
        'badge_number': [101, 102, 103],
        'hours_worked': [40, 35, 42],
        'name': ['Alice', 'Bob', 'Charlie']
    },
    categorical={
        'employee_id': False,    # Treat as continuous (unusual but allowed)
        'badge_number': True,    # Treat as categorical (default anyway)
        'hours_worked': False    # Treat as continuous count
    }
)

print(f"\nExplicitly specified categorical:")
for col, is_cat in df3.categorical.items():
    status = "Categorical" if is_cat else "Non-categorical"
    print(f"  • {col:15s} → {status}")

print("\n" + "="*80)
print("8. USE CASE: DATA ANALYSIS")
print("="*80)

print("\nWith proper categorical tracking, you can:")
print("  ✓ Apply appropriate statistical methods")
print("  ✓ Choose correct visualization types")
print("  ✓ Validate operations before execution")
print("  ✓ Optimize algorithms for categorical data")

# Example: Separate analysis for categorical vs continuous
print(f"\nContinuous variables (can compute statistics):")
for col in df2.get_non_categorical_columns():
    values = df2[col]
    if hasattr(values, 'mean'):
        print(f"  • {col:20s} mean = {values.mean():.2f}")
    else:
        arr = jnp.array(values) if not isinstance(values, jnp.ndarray) else values
        print(f"  • {col:20s} mean = {arr.mean():.2f}")

print(f"\nCategorical variables (can count unique values):")
for col in df2.get_categorical_columns():
    values = df2[col]
    if isinstance(values, list):
        unique = len(set(values))
    else:
        unique = len(jnp.unique(values))
    print(f"  • {col:20s} unique = {unique}")

print("\n" + "="*80)
print("9. IMMUTABILITY PRESERVED")
print("="*80)

print("\nOriginal DataFrame unchanged:")
print(f"  df.is_categorical('age'):  {df.is_categorical('age')}")
print(f"\nModified DataFrame different:")
print(f"  df2.is_categorical('age'): {df2.is_categorical('age')}")

print("\n" + "="*80)
print("✅ DEMO COMPLETE")
print("="*80)
print("\nKey Takeaways:")
print("  1. dtypes now returns element types (not 'list[type]')")
print("  2. Categorical status is automatically inferred")
print("  3. Float columns: always non-categorical")
print("  4. String columns: always categorical")
print("  5. Int/bool columns: categorical by default, but can be changed")
print("  6. Use as_categorical() and as_non_categorical() to modify")
print("  7. Query with is_categorical(), get_categorical_columns()")
print("  8. All operations return new DataFrames (immutable)")
print("="*80)
