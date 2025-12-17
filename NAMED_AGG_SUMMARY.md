# Named Aggregation Feature Summary

## Overview
The `GroupBy.agg()` method now supports explicit naming of custom aggregation functions using tuple syntax: `(name, function)`.

## Feature Details

### Syntax
```python
df.group_by('column').agg({
    'value_col': ('custom_name', aggregation_function)
})
```

### Supported Formats
The `agg()` method now accepts:
1. **Built-in strings**: `'sum'`, `'mean'`, `'std'`, `'min'`, `'max'`, `'count'`
2. **Callables** (auto-named): `jnp.median`, `custom_func`, `lambda x: ...`
3. **Named tuples** (explicit naming): `('p90', lambda x: jnp.percentile(x, 90))`
4. **Lists**: Any combination of the above

### Type Signature
```python
def agg(
    self,
    agg_dict: Dict[str, Union[
        str,                              # Built-in string
        Callable,                         # Unnamed function
        Tuple[str, Callable],             # Named tuple ✨ NEW
        List[Union[str, Callable, Tuple[str, Callable]]]
    ]]
) -> DataFrame:
```

## Examples

### Basic Named Function
```python
df.group_by('category').agg({
    'value': ('range', lambda x: jnp.max(x) - jnp.min(x))
})
# Result columns: 'category', 'value'  (single agg, no suffix)
```

### Multiple Named Functions
```python
df.group_by('group').agg({
    'score': [
        ('p25', lambda x: jnp.percentile(x, 25)),
        ('median', jnp.median),
        ('p75', lambda x: jnp.percentile(x, 75))
    ]
})
# Result columns: 'group', 'score_p25', 'score_median', 'score_p75'
```

### Mixing All Types
```python
df.group_by('category').agg({
    'value': [
        'mean',                                      # Built-in string
        jnp.median,                                  # Unnamed callable
        ('p90', lambda x: jnp.percentile(x, 90))    # Named tuple
    ]
})
# Result columns: 'category', 'value_mean', 'value_median', 'value_p90'
```

### Financial Metrics
```python
df.group_by('stock').agg({
    'price': [
        ('low', jnp.min),
        ('high', jnp.max),
        ('avg', jnp.mean),
        ('volatility', jnp.std)
    ]
})
# Result columns: 'stock', 'price_low', 'price_high', 'price_avg', 'price_volatility'
```

## Benefits

### ✅ Use Named Tuples When:
- You want clear, readable column names
- The function's `__name__` is unclear (e.g., lambda → `<lambda>`)
- You want consistent naming across multiple columns
- Computing domain-specific metrics (e.g., `p90`, `cv`, `snr`)
- You want shorter names than the function's `__name__`

### ❌ Don't Need Named Tuples When:
- Using built-in strings (`'mean'`, `'sum'`, etc.)
- Function has a good `__name__` (e.g., `jnp.median` → `'median'`)
- Auto-naming is sufficient

## Validation

### Tuple Validation Rules
1. **Length**: Must be exactly 2 elements
   ```python
   ('name', func)  # ✅ Valid
   ('name', func, extra)  # ❌ ValueError
   ```

2. **First Element**: Must be a string
   ```python
   ('p90', func)  # ✅ Valid
   (123, func)    # ❌ TypeError
   ```

3. **Second Element**: Must be callable
   ```python
   ('name', lambda x: x)  # ✅ Valid
   ('name', 'not_func')   # ❌ TypeError
   ```

## Column Naming Rules

### Single Aggregation
When using a single aggregation function, the column keeps its original name:
```python
df.group_by('group').agg({'value': ('custom', func)})
# Result column: 'value' (not 'value_custom')
```

### Multiple Aggregations
When using multiple aggregations, each gets a suffix with the function name:
```python
df.group_by('group').agg({
    'value': [
        'mean',
        ('p90', func)
    ]
})
# Result columns: 'value_mean', 'value_p90'
```

## Implementation Details

### Processing Order
1. Check if tuple → validate and apply with explicit name
2. Check if callable → apply with auto-generated name
3. Check if string → apply built-in aggregation

### Error Messages
- **Wrong tuple length**: `"Tuple aggregation must be (name, function), got length N"`
- **Invalid name type**: `"First element of tuple must be a string name, got <type>"`
- **Invalid function type**: `"Second element of tuple must be a callable, got <type>"`

## Test Coverage

### Test Files
- `tests/test_named_agg.py`: 18 tests for tuple syntax
- `tests/test_custom_agg.py`: 21 tests for general custom aggregations

### Test Categories
1. **Basic functionality** (5 tests)
   - Single named function
   - Multiple named functions
   - Mixing named, unnamed, and built-in

2. **JAX compatibility** (2 tests)
   - Named JAX functions
   - JIT-compiled functions

3. **Edge cases** (5 tests)
   - Wrong tuple length
   - Invalid name type
   - Invalid function type
   - Empty names
   - Special characters

4. **Real-world examples** (3 tests)
   - Financial metrics
   - Survey analysis
   - Sensor quality metrics

5. **Complex scenarios** (2 tests)
   - Mixing all aggregation types
   - Single named tuple (no suffix)

### All Tests Passing ✅
- **Named aggregations**: 18/18 passing
- **Custom aggregations**: 21/21 passing
- **Total**: 39/39 passing

## Demo Files

### demo_named_agg.py
10 comprehensive examples demonstrating:
1. Basic named functions
2. Multiple named functions
3. Financial metrics
4. Mixing aggregation types
5. Sensor quality metrics
6. Survey analysis
7. Multiple columns
8. Complex custom metrics
9. Single function naming
10. Named vs unnamed comparison

Run with:
```bash
python demo_named_agg.py
```

## Files Modified

### src/jaxframe/dataframe.py
- Updated `GroupBy.agg()` method signature with `Tuple[str, Callable]`
- Added tuple validation logic
- Added explicit naming support
- Updated docstring with tuple examples

### tests/test_named_agg.py
- Created comprehensive test suite for tuple syntax
- 18 tests covering all use cases

### tests/test_custom_agg.py
- Updated error message regex to include "tuples"

### demo_named_agg.py
- Created 10-example demo file
- Shows practical use cases and best practices

## Backward Compatibility

✅ **Fully backward compatible**: All existing code continues to work:
- Built-in strings: `'mean'`, `'sum'`, etc.
- Callables with auto-naming: `jnp.median`, custom functions
- Lists of the above

## JAX Compatibility

✅ **Fully JAX compatible**:
- Works with JIT-compiled functions
- Works with JAX transformations (vmap, grad)
- Preserves JAX array types

## Summary

The named tuple syntax `(name, function)` provides explicit control over aggregation result column names, making code more readable and maintainable, especially for:
- Complex statistical functions
- Domain-specific metrics  
- Lambda functions
- Consistent naming across multiple columns

This enhancement maintains full backward compatibility while adding powerful new functionality for explicit naming control.
