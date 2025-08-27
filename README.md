# jaxframe

A simple immutable DataFrame library that supports both Python lists and NumPy arrays as column data.

## Installation

```bash
pip install -e .
```

## Usage

```python
from jaxframe import DataFrame
import numpy as np

# Create DataFrame with mixed types
data = {
    'names': ['Alice', 'Bob', 'Charlie'],    # Python list
    'ages': np.array([25, 30, 35]),          # NumPy array
    'scores': [85.5, 92.0, 78.5]            # Python list
}

df = DataFrame(data)
print(df)
print(f"Column types: {df.column_types}")
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
