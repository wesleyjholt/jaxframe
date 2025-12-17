# Categorical Column Tracking

JAXFrame now includes automatic tracking of which columns are categorical (discrete/enumerable) vs continuous.

## Overview

The DataFrame automatically infers categorical status based on column dtype:

- **Float columns**: Always non-categorical (cannot be changed)
- **String columns**: Always categorical (cannot be changed)  
- **Int columns**: Categorical by default (can be changed)
- **Bool columns**: Categorical by default (can be changed)

## API

### Properties and Query Methods

```python
# Get categorical status for all columns
df.categorical  # Returns: Dict[str, bool]

# Check if specific column is categorical
df.is_categorical('column_name')  # Returns: bool

# Get lists of categorical/non-categorical columns
df.get_categorical_columns()      # Returns: List[str]
df.get_non_categorical_columns()  # Returns: List[str]
```

### Modification Methods

```python
# Mark columns as categorical
df2 = df.as_categorical('user_id')
df2 = df.as_categorical(['user_id', 'product_id'])

# Mark columns as non-categorical
df2 = df.as_non_categorical('count')
df2 = df.as_non_categorical(['count', 'quantity'])
```

### Constructor Override

```python
# Explicitly specify categorical status in constructor
df = DataFrame(
    {
        'id': [1, 2, 3],
        'count': [10, 20, 30],
        'name': ['A', 'B', 'C']
    },
    categorical={'id': False, 'count': True}
)
```

## Examples

### Basic Usage

```python
from jaxframe import DataFrame
import jax.numpy as jnp

df = DataFrame({
    'user_id': [1, 2, 3],              # Int - categorical by default
    'name': ['Alice', 'Bob', 'Charlie'], # String - always categorical
    'age': [25, 30, 35],                # Int - categorical by default
    'salary': [50000.0, 60000.0, 70000.0]  # Float - always non-categorical
})

print(df.categorical)
# Output: {'user_id': True, 'name': True, 'age': True, 'salary': False}

# Get categorical columns
print(df.get_categorical_columns())
# Output: ['user_id', 'name', 'age']

# Get non-categorical columns  
print(df.get_non_categorical_columns())
# Output: ['salary']
```

### Making Int Column Non-Categorical

```python
# Age is more naturally a continuous variable
df2 = df.as_non_categorical('age')

print(df2.categorical)
# Output: {'user_id': True, 'name': True, 'age': False, 'salary': False}
```

### Multiple Columns

```python
# Mark multiple int columns as non-categorical
df2 = df.as_non_categorical(['user_id', 'age'])

print(df2.get_non_categorical_columns())
# Output: ['user_id', 'age', 'salary']
```

### Validation

```python
# Float columns cannot be categorical
try:
    df.as_categorical('salary')
except ValueError as e:
    print(e)
    # Output: Cannot mark float-like column 'salary' as categorical.
    #         Float columns must be non-categorical. Detected dtype: float

# String columns cannot be non-categorical
try:
    df.as_non_categorical('name')
except ValueError as e:
    print(e)
    # Output: Cannot mark string-like column 'name' as non-categorical.
    #         String columns must be categorical. Detected dtype: str
```

### Constructor Override

```python
# Specify categorical status from the start
df = DataFrame(
    {
        'employee_id': [1, 2, 3],
        'department_id': [10, 20, 30],
        'salary': [50000.0, 60000.0, 70000.0]
    },
    categorical={
        'employee_id': False,   # Treat as continuous
        'department_id': True   # Treat as categorical
    }
)

print(df.categorical)
# Output: {'employee_id': False, 'department_id': True, 'salary': False}
```

## Design Rationale

### Why These Rules?

1. **Float columns are always non-categorical**
   - Float values are inherently continuous
   - Treating floats as categorical would be misleading
   - If you need categorical floats, convert to int or string first

2. **String columns are always categorical**
   - Strings represent discrete categories/labels
   - Continuous operations on strings don't make sense
   - This prevents accidental misuse

3. **Int/bool columns are flexible**
   - Integers can represent either categorical IDs or continuous counts
   - Booleans are technically categorical but can be treated as 0/1
   - User can choose based on domain knowledge

### Use Cases

This categorical tracking is useful for:

- **Data validation**: Ensure operations are appropriate for data type
- **Visualization**: Choose appropriate plot types
- **Statistical analysis**: Select correct statistical methods
- **Machine learning**: Apply proper encoding strategies
- **Optimization**: Enable specialized algorithms for categorical data

## Type Inference

The system inspects column data to determine the base type:

### For Arrays (NumPy/JAX)
```python
import numpy as np
import jax.numpy as jnp

df = DataFrame({
    'np_int': np.array([1, 2, 3]),      # Categorical by default
    'np_float': np.array([1.0, 2.0, 3.0]),  # Non-categorical
    'jax_int': jnp.array([1, 2, 3]),    # Categorical by default
    'jax_float': jnp.array([1.0, 2.0, 3.0])  # Non-categorical
})
```

### For Lists
```python
df = DataFrame({
    'int_list': [1, 2, 3],          # Categorical by default
    'float_list': [1.0, 2.0, 3.0],  # Non-categorical
    'str_list': ['A', 'B', 'C'],    # Always categorical
    'bool_list': [True, False, True]  # Categorical by default
})
```

## Immutability

All modifications return new DataFrames:

```python
df1 = DataFrame({'id': [1, 2, 3]})
df2 = df1.as_non_categorical('id')

print(df1.is_categorical('id'))  # True (unchanged)
print(df2.is_categorical('id'))  # False (new DataFrame)
```

## Integration with DataFrame Operations

Categorical tracking is preserved through operations:

```python
df1 = DataFrame({'id': [1, 2, 3], 'value': [10.0, 20.0, 30.0]})
df2 = df1.as_non_categorical('id').with_name('processed')

# Categorical flags are preserved
assert df2.is_categorical('id') == False
assert df2.name == 'processed'
```
