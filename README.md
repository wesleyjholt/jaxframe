# jaxframe

A simple, jax-compatible dataframe library. 

`jaxframe` provides a convenient way to differentiate functions that perform (simple) relational database-like operations (add, join, etc.). 
All dataframes are immutable and all data transformations are implemented as pure functions.

## Installation

```bash
git clone git@github.com:wesleyjholt/jaxframe.git
cd jaxframe
pip install -e .
```
or
```bash
pip install git+https://github.com/wesleyjholt/jaxframe.git
```

## Usage

```python
from jaxframe import DataFrame
import numpy as np
import jax.numpy as jnp

# Create DataFrame with mixed types
data = {
    'names': ['Alice', 'Bob', 'Charlie'],    # Python list
    'ages': np.array([25, 30, 35]),          # NumPy array
    'scores': jnp.array([85.5, 92.0, 78.5])  # JAX array
}

df = DataFrame(data)
print(df)
print(f"Column types: {df.column_types}")

# DataFrame(3 rows, 3 columns)
# Columns: names, ages, scores
#   [0]: {'names': 'Alice', 'ages': '25', 'scores': '85.5'}
#   [1]: {'names': 'Bob', 'ages': '30', 'scores': '92.0'}
#   [2]: {'names': 'Charlie', 'ages': '35', 'scores': '78.5'}
# Column types: {'names': 'list', 'ages': 'array', 'scores': 'jax_array'}
```

## Testing

Run tests using pytest:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_dataframe.py

# Run with verbose output
pytest -v

# Run using the convenience script
./run_tests.sh
```

## Features

- **Immutable**: DataFrames cannot be modified after creation
- **Type preservation**: Lists stay as lists, arrays stay as arrays
- **Length validation**: All columns must have the same length
- **Mixed types**: Supports both Python lists and NumPy arrays as columns
- **Type-aware operations**: Column selection, equality, etc. preserve original types
