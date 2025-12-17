# Removal of String Shortcuts in Aggregation - Summary

## Overview
String shortcuts for aggregation functions (`'mean'`, `'sum'`, `'std'`, `'min'`, `'max'`, `'count'`) have been **completely removed** from `GroupBy.agg()`. Users must now use either:
1. **Callables**: Direct JAX functions like `jnp.mean`, `jnp.sum`, custom functions
2. **Named tuples**: `(name, function)` for explicit control over result column names

## Changes Made

### 1. Updated `GroupBy.agg()` Method
**File**: `src/jaxframe/dataframe.py`

#### Type Signature (Before)
```python
def agg(self, agg_dict: Dict[str, Union[
    str,                              # ❌ REMOVED
    List[str],                        # ❌ REMOVED  
    Callable,
    Tuple[str, Callable],
    List[Union[str, Callable, Tuple[str, Callable]]]
]]) -> 'DataFrame':
```

#### Type Signature (After)
```python
def agg(self, agg_dict: Dict[str, Union[
    Callable,                         # ✅ Direct functions
    Tuple[str, Callable],             # ✅ Named tuples
    List[Union[Callable, Tuple[str, Callable]]]
]]) -> 'DataFrame':
```

#### Implementation Changes
- **Removed**: All string-based aggregation logic (`if agg_func_lower == 'sum'`, etc.)
- **Removed**: Special handling for built-in string names
- **Added**: Clear error message when strings are passed:
  ```
  "Aggregation function must be a callable or tuple (name, callable), got str.
   String shortcuts like 'mean', 'sum', etc. are no longer supported.
   Use jnp.mean, jnp.sum, or tuples like ('mean', jnp.mean) instead."
  ```

### 2. Updated Tests
**Files**:
- `tests/test_group_by.py`
- `tests/test_groupby_apply.py`  
- `tests/test_custom_agg.py`
- `tests/test_named_agg.py`

**Changes**:
- Added helper aggregation functions at the top of each test file:
  ```python
  def agg_sum(x): return jnp.sum(x)
  def agg_mean(x): return jnp.mean(x)
  def agg_std(x): return jnp.std(x)
  def agg_min(x): return jnp.min(x)
  def agg_max(x): return jnp.max(x)
  def agg_count(x): return jnp.array(len(x), dtype=x.dtype)
  ```
- Replaced all string shortcuts with named tuples:
  - `'mean'` → `('mean', agg_mean)`
  - `'sum'` → `('sum', agg_sum)`
  - `['mean', 'max']` → `[('mean', agg_mean), ('max', agg_max)]`
- Updated error assertion tests to expect `TypeError` instead of `ValueError`

**Test Results**: **80/80 tests passing** ✅

## Migration Guide

### Old Syntax (❌ No Longer Supported)
```python
# Single string
df.group_by('category').agg({'value': 'mean'})

# List of strings
df.group_by('category').agg({'value': ['mean', 'max']})

# Mixed strings and functions
df.group_by('category').agg({'value': ['mean', custom_func]})
```

### New Syntax (✅ Required)
```python
# Single callable
df.group_by('category').agg({'value': jnp.mean})

# Single named tuple
df.group_by('category').agg({'value': ('mean', jnp.mean)})

# List of named tuples
df.group_by('category').agg({'value': [
    ('mean', jnp.mean),
    ('max', jnp.max)
]})

# Mixed named tuples and unnamed callables
df.group_by('category').agg({'value': [
    ('mean', jnp.mean),
    jnp.median,  # Auto-named as 'median'
    custom_func   # Auto-named from __name__
]})
```

## Column Naming Behavior

### Single Aggregation
When applying a single aggregation function, the column keeps its original name:
```python
df.group_by('id').agg({'value': ('mean', jnp.mean)})
# Result columns: ('id', 'value')  ← No suffix added
```

### Multiple Aggregations
When applying multiple aggregations, each gets a suffix:
```python
df.group_by('id').agg({'value': [
    ('mean', jnp.mean),
    ('std', jnp.std)
]})
# Result columns: ('id', 'value_mean', 'value_std')
```

### Mixed Named and Unnamed
```python
df.group_by('id').agg({'value': [
    ('mean', jnp.mean),     # Uses explicit name 'mean'
    jnp.median,              # Uses __name__ → 'median'
    lambda x: jnp.max(x)    # Uses __name__ → '<lambda>' or 'custom_0'
]})
# Result columns: ('id', 'value_mean', 'value_median', 'value_<lambda>')
```

## Rationale for Removal

### Benefits
1. **Explicit is better than implicit**: Users see exactly which JAX function is being used
2. **Consistency**: No special cases for built-in vs custom functions
3. **Flexibility**: All JAX functions available, not just a predefined set
4. **Type safety**: Clearer type signatures without string unions
5. **Discoverability**: IDEs can autocomplete JAX functions
6. **Performance**: No string parsing or lookup overhead

### Trade-offs
- **Migration effort**: Existing code needs to be updated
- **Verbosity**: `('mean', jnp.mean)` is longer than `'mean'`
  - **Solution**: Users can define their own helper functions or aliases

## Example: User's Test Case

```python
import jax.numpy as jnp
from jaxframe import DataFrame

height_meas_table = DataFrame({
    'id_height_meas': ['01', '02', '03'],
    'id_person': ['01', '01', '02'],
    'height_meas': jnp.array([55.0, 55.5, 60.0])
})

# ❌ Throws TypeError
height_meas_table.group_by('id_person').agg({'height_meas': 'mean'})

# ❌ Throws TypeError
height_meas_table.group_by('id_person').agg({'height_meas': ['mean', 'max']})

# ✅ Works - Creates column 'height_meas' (no suffix for single agg)
height_meas_table.group_by('id_person').agg({
    'height_meas': ('mean', jnp.mean)
})

# ✅ Works - Creates columns 'height_meas_mean' and 'height_meas_std'
height_meas_table.group_by('id_person').agg({
    'height_meas': [('mean', jnp.mean), ('std', jnp.std)]
})
```

## Documentation Updates Needed

1. **README.md**: Update GroupBy examples to use new syntax
2. **API documentation**: Remove references to string shortcuts
3. **Tutorial notebooks**: Update all aggregation examples
4. **Demo files**: Update `demo_group_by.py`, `demo_groupby_apply.py`, `demo_custom_agg.py`

## Files Modified

### Core Implementation
- `src/jaxframe/dataframe.py`: Removed string shortcuts, updated type hints

### Tests (80/80 passing)
- `tests/test_group_by.py`: 22 tests updated
- `tests/test_groupby_apply.py`: 19 tests updated  
- `tests/test_custom_agg.py`: 21 tests updated
- `tests/test_named_agg.py`: 18 tests updated

### Helper Scripts
- `test_no_string_shortcuts.py`: Comprehensive test of new behavior
- `update_tests_for_no_strings.py`: Automated test migration script

## Summary

String shortcuts have been **completely removed** from aggregation functions. Users must now use:
- **Callables**: `jnp.mean`, `jnp.sum`, custom functions
- **Named tuples**: `('name', function)` for explicit naming

This change makes the API more explicit, consistent, and JAX-native while maintaining full backward compatibility for users already using callables. The migration is straightforward: replace strings with tuples or direct functions.

**All tests passing**: 80/80 ✅
