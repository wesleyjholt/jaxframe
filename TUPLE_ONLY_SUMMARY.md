# Tuple-Only Aggregation Summary

## Change
Aggregation functions in `GroupBy.agg()` now **ONLY** accept tuples of `(name, callable)`.

## Before (No Longer Supported)
```python
# Bare callables - NO LONGER ALLOWED
df.group_by('cat').agg({'value': jnp.mean})
df.group_by('cat').agg({'value': [jnp.mean, jnp.std]})

# String shortcuts - NO LONGER ALLOWED  
df.group_by('cat').agg({'value': 'mean'})
df.group_by('cat').agg({'value': ['mean', 'std']})
```

## After (Required)
```python
# Must use tuples (name, callable)
df.group_by('cat').agg({'value': ('mean', jnp.mean)})
df.group_by('cat').agg({'value': [('mean', jnp.mean), ('std', jnp.std)]})
```

## Type Signature
```python
def agg(
    self,
    agg_dict: Dict[str, Union[
        Tuple[str, Callable],              # Single tuple
        List[Tuple[str, Callable]]         # List of tuples
    ]]
) -> 'DataFrame':
```

## Column Naming
- **Single aggregation**: Column gets `_{name}` suffix
  - `{'value': ('mean', jnp.mean)}` → column named `'value_mean'`
- **Multiple aggregations**: Columns get `_{name}` suffix
  - `{'value': [('mean', jnp.mean), ('std', jnp.std)]}` → columns `'value_mean'`, `'value_std'`

## Rationale
1. **Explicit naming**: Forces users to provide meaningful names
2. **No magic**: No auto-naming from `__name__` attribute  
3. **Consistency**: Single format for all aggregations
4. **Clarity**: Clear what each aggregation is named in results

## Examples
```python
import jax.numpy as jnp
from jaxframe import DataFrame

df = DataFrame({
    'category': ['A', 'A', 'B', 'B'],
    'value': jnp.array([1, 2, 3, 4])
})

# Single aggregation - no suffix
df.group_by('category').agg({
    'value': ('mean', jnp.mean)
})
# Result columns: ('category', 'value')

# Multiple aggregations - with suffix
df.group_by('category').agg({
    'value': [
        ('mean', jnp.mean),
        ('std', jnp.std),
        ('p90', lambda x: jnp.percentile(x, 90))
    ]
})
# Result columns: ('category', 'value_mean', 'value_std', 'value_p90')
```

## Migration from Previous Version
Replace all bare callables with tuples:
- `jnp.mean` → `('mean', jnp.mean)`
- `jnp.sum` → `('sum', jnp.sum)`
- `custom_func` → `('custom', custom_func)` or `('descriptive_name', custom_func)`
- `lambda x: ...` → `('name', lambda x: ...)`

## Test Status
- `test_only_tuples.py`: ✅ All passing (demonstrates new behavior)
- `test_named_agg.py`: ✅ Most tests passing (already used tuples)
- `test_group_by.py`: ✅ Using helper functions with tuples
- `test_custom_agg.py`: ❌ Needs update (uses bare callables)
