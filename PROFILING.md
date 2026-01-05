# Profiling Jaxframe Transforms

This document describes the profiling instrumentation available in jaxframe to help diagnose performance issues during JAX compilation.

## Quick Start

Enable profiling by setting environment variables:

```bash
# Enable profiling for first 20 calls (default)
export JAXFRAME_PROFILE_TRANSFORM=1
python your_script.py

# Enable profiling for first 100 calls
export JAXFRAME_PROFILE_TRANSFORM=1
export JAXFRAME_PROFILE_LIMIT=100
python your_script.py
```

## Environment Variables

### `JAXFRAME_PROFILE_TRANSFORM`

Controls whether profiling is enabled. Set to one of:
- `1`, `true`, `True`, `yes`, `YES` - Enable profiling
- Any other value or unset - Disable profiling (default)

### `JAXFRAME_PROFILE_LIMIT`

Maximum number of profiled function calls to output. Default is 20.

This prevents excessive output during inference when the same function is called many times. Set to a higher value to see more calls.

## What Gets Profiled

When profiling is enabled, timing information is captured for:

1. **Pivot operations** (`pivot_sparse` / `long_to_wide_masked`)
   - PivotStructure creation (Python)
   - Value extraction (JAX)
   - Structure application (JAX scatter/sort)
   - Output DataFrame construction (Python)

2. **Unpivot operations** (`unpivot_sparse` / `wide_to_long_masked`)
   - UnpivotStructure creation (Python)
   - Wide value extraction (JAX)
   - Structure application (JAX gather)
   - Long DataFrame construction (Python)

3. **Masked array conversions** (`to_masked_array` / `wide_df_to_masked_array`)
   - Column finding (Python)
   - Value extraction (JAX)

## Profiling Output Format

Each profiled function call produces output like:

```
[JAXFRAME-TIMING] pivot_jax: call #1, traced=True, n_rows=100, sort_within_id=False
[JAXFRAME-TIMING] pivot/PivotStructure: 0.003s
[JAXFRAME-TIMING] pivot/extract_values: 0.001s
[JAXFRAME-TIMING] pivot/apply_structure: 0.012s
[JAXFRAME-TIMING] pivot/build_output: 0.002s
```

Fields:
- **Function name**: Which transform function was called
- **Call number**: Sequential call counter for this function
- **traced**: Whether JAX is currently tracing (compilation)
- **n_rows**: Number of rows being processed
- **Subphase timings**: Time spent in each subphase

## Tracing Detection

The profiling automatically detects whether values are being traced by JAX during JIT compilation:

- `traced=True`: Function is being called during JAX JIT compilation (tracing)
- `traced=False`: Function is being called with concrete (non-traced) values

This is useful because compilation happens during the first call to a JIT-compiled function, and the profiling will show which transforms contribute to compilation time.

## Call Limiting

To avoid overwhelming output during loops, profiling automatically limits output to the first N calls (default 20). When the limit is reached, you'll see:

```
[JAXFRAME-TIMING] pivot_jax: profiling limit reached (20 calls), suppressing further output
```

You can increase the limit with `JAXFRAME_PROFILE_LIMIT`:

```bash
export JAXFRAME_PROFILE_LIMIT=100
```

## Example: Investigating Slow Compilation

Suppose you have a model that's slow to compile. Here's how to use profiling:

```python
import jax
import jax.numpy as jnp
from jaxframe import DataFrame
from jaxframe.transform import pivot_sparse

@jax.jit
def my_model(data):
    # Create DataFrame
    df = DataFrame({
        'id': ...,
        'time': ...,
        'value': data
    })
    
    # Pivot operation
    wide = pivot_sparse(df, index='id', value='value', on='time')
    
    # ... rest of model ...
    return result

# First call will trace and show profiling
result = my_model(jnp.array([1.0, 2.0, 3.0, ...]))
```

With profiling enabled, you'll see output like:

```
[JAXFRAME-TIMING] pivot_jax: call #1, traced=True, n_rows=1000, sort_within_id=False
[JAXFRAME-TIMING] pivot/PivotStructure: 0.125s
[JAXFRAME-TIMING] pivot/extract_values: 0.003s
[JAXFRAME-TIMING] pivot/apply_structure: 2.456s
[JAXFRAME-TIMING] pivot/build_output: 0.012s
```

This tells you that most time is spent in `apply_structure` (the JAX scatter/sort operations).

## Demo Script

Run the included demo script to see profiling in action:

```bash
# Without profiling (default)
python demo_profiling.py

# With profiling enabled
JAXFRAME_PROFILE_TRANSFORM=1 python demo_profiling.py

# With custom limit
JAXFRAME_PROFILE_TRANSFORM=1 JAXFRAME_PROFILE_LIMIT=5 python demo_profiling.py
```

## Performance Impact

The profiling infrastructure has minimal overhead when disabled (just a function call and environment variable check). When enabled, timing uses Python's `perf_counter` which has microsecond resolution with minimal overhead.

## Integration with Other Profiling Tools

This profiling is complementary to:

- **JAX profiling** (`JAX_LOG_COMPILES=1`): Shows what JAX is compiling
- **XLA profiling** (`XLA_FLAGS=--xla_dump_to=...`): Shows XLA HLO graphs
- **Python profilers** (cProfile, py-spy): Show overall Python execution time

Use jaxframe profiling to understand time spent in transform operations, then use other tools to dig deeper into specific bottlenecks.

## Troubleshooting

### Profiling output not showing

Check that:
1. `JAXFRAME_PROFILE_TRANSFORM=1` is set
2. The transform operation is actually being called
3. You haven't exceeded `JAXFRAME_PROFILE_LIMIT` (default 20 calls)
4. The transform is using JAX arrays (profiling targets JAX-compiled paths)

### Too much output

Reduce `JAXFRAME_PROFILE_LIMIT`:

```bash
export JAXFRAME_PROFILE_LIMIT=5
```

### Want to see all calls

Set a very high limit:

```bash
export JAXFRAME_PROFILE_LIMIT=999999
```

## Testing

Run the profiling test suite:

```bash
pytest tests/test_profiling.py -v
```

This verifies that:
- Profiling can be enabled/disabled
- Call limiting works correctly
- Profiling doesn't break functionality
- Output is captured correctly
