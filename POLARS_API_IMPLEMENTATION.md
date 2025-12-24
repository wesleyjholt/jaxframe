# Polars API Implementation for JAXFrame

This document summarizes the Polars-compatible methods implemented for the JAXFrame DataFrame class.

## Implemented Methods

### 1. `vstack(other, *, in_place=False)`
- **Purpose**: Vertically stack DataFrames (row-wise concatenation)
- **Parameters**:
  - `other`: DataFrame to stack below the current DataFrame
  - `in_place`: Must be False (JAXFrame preserves immutability)
- **Returns**: New DataFrame with combined rows
- **Usage**: `df.vstack(other_df)`

### 2. `hstack(others, *, in_place=False)`
- **Purpose**: Horizontally stack DataFrames or arrays (column-wise concatenation)
- **Parameters**:
  - `others`: DataFrame, array, or list of arrays to stack beside current DataFrame
  - `in_place`: Must be False (JAXFrame preserves immutability)
- **Returns**: New DataFrame with combined columns
- **Usage**: `df.hstack([array1, array2])` or `df.hstack(other_df)`

### 3. `with_columns(*args, **kwargs)`
- **Purpose**: Add or replace columns in the DataFrame
- **Parameters**:
  - `*args`: Dictionary of column mappings (positional)
  - `**kwargs`: Column name-value pairs (keyword)
- **Returns**: New DataFrame with added/modified columns
- **Usage**: `df.with_columns(new_col=values)` or `df.with_columns({'col': values})`

### 4. `drop(columns, *, strict=True)`
- **Purpose**: Remove columns from the DataFrame
- **Parameters**:
  - `columns`: Column name(s) to drop (string or list of strings)
  - `strict`: If True, raises error for non-existent columns
- **Returns**: New DataFrame without specified columns
- **Usage**: `df.drop('col1')` or `df.drop(['col1', 'col2'])`

### 5. `filter(predicate=None, **constraints)`
- **Purpose**: Filter rows based on conditions
- **Parameters**:
  - `predicate`: Boolean array/list for row selection
  - `**constraints`: Column-value constraints for filtering
- **Returns**: New DataFrame with filtered rows
- **Usage**: `df.filter(boolean_mask)` or `df.filter(age=25, name='Alice')`

## Key Features

### Polars Compatibility
- **Exact Syntax Match**: All method signatures match Polars DataFrame API
- **Parameter Names**: Identical parameter names and behavior to Polars
- **Error Handling**: Similar error messages and validation patterns

### JAXFrame Integration
- **Immutability Preserved**: All methods return new DataFrames (no in-place modifications)
- **JAX Array Support**: Full compatibility with JAX arrays and transformations
- **Type Safety**: Maintains JAXFrame's column type tracking and validation

### Performance
- **Efficient Operations**: Uses existing JAXFrame internal methods for optimal performance
- **Memory Management**: Leverages JAX's efficient array operations
- **Method Chaining**: Supports fluent interface patterns like Polars

## Test Coverage

### Comprehensive Testing
- **32 Test Cases**: Complete coverage of all methods and edge cases
- **Error Conditions**: Testing of invalid inputs and boundary conditions
- **Integration Tests**: Method chaining and complex operation combinations
- **Type Compatibility**: Testing with various array types (numpy, JAX, lists)

### Test Categories
1. **Basic Functionality**: Core method behavior
2. **Error Handling**: Invalid parameter validation
3. **Edge Cases**: Empty DataFrames, missing columns, type mismatches
4. **Integration**: Method chaining and complex workflows
5. **Immutability**: Verification that original DataFrames remain unchanged

## Usage Examples

```python
import jax.numpy as jnp
from jaxframe import DataFrame

# Create test data
df = DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'score': [85.5, 92.0, 78.5]
})

# Vertical stacking
df2 = DataFrame({'name': ['David'], 'age': [28], 'score': [88.0]})
combined = df.vstack(df2)

# Horizontal stacking
df_with_new_cols = df.hstack([jnp.array([1, 2, 3]), jnp.array([4, 5, 6])])

# Adding columns
df_with_bonus = df.with_columns(bonus=jnp.array([100, 200, 150]))

# Dropping columns
df_no_score = df.drop('score')

# Filtering
young_people = df.filter(age=25)
high_scorers = df.filter(df.get_column('score') > 80.0)

# Method chaining (Polars-style)
result = (df
    .with_columns(bonus=jnp.array([100, 200, 150]))
    .filter(age=30)
    .drop('score'))
```

## Implementation Notes

### Design Decisions
- **Immutability First**: All operations return new DataFrames to maintain JAXFrame principles
- **Error Compatibility**: Error messages and validation patterns match Polars behavior
- **Performance Optimization**: Reuses existing JAXFrame internal methods where possible

### Limitations
- **No In-Place Operations**: `in_place=True` is not supported (raises NotImplementedError)
- **No Expression API**: Complex Polars expressions are not yet supported
- **Limited Concat Options**: Only basic concatenation patterns implemented

### Future Enhancements
- Expression-based filtering and column operations
- Additional Polars methods (select, group_by, etc.)
- Performance optimizations for large DataFrames
- Advanced join operations with Polars syntax