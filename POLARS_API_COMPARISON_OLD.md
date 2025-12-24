# JAXFrame vs Polars API Feature Comparison

*Generated: October 7, 2025*

This document provides a comprehensive comparison between JAXFrame and Polars DataFrame features, identifying similar functionality and gaps for potential API alignment.

## Core DataFrame Features

| **JAXFrame Feature** | **JAXFrame Method/Property** | **Similar Polars Feature** | **Polars Method/Property** | **Notes** |
|----------------------|-------------------------------|----------------------------|----------------------------|-----------|
| **DataFrame Creation** | `DataFrame(data)` | DataFrame Constructor | `pl.DataFrame(data)` | Both support dict-based creation |
| **Fast Constructors** | `DataFrame.from_jax_arrays()` | Constructor with schema | `pl.DataFrame(data, schema=...)` | JAXFrame optimized for JAX, Polars for Arrow |
| **Fast Constructors** | `DataFrame.from_numpy_arrays()` | Constructor from numpy | `pl.DataFrame(data)` | Both handle numpy arrays |
| **Fast Constructors** | `DataFrame.from_lists()` | Constructor from lists | `pl.DataFrame(data)` | Both handle list data |
| **Shape Information** | `df.shape` | Shape property | `df.shape` | Identical interface |
| **Length** | `len(df)` | Length | `len(df)` | Identical interface |
| **Column Names** | `df.columns` | Column names | `df.columns` | Identical interface |
| **Data Types** | `df.dtypes` | Data types | `df.dtypes` | Both are properties, JAXFrame returns dict, Polars returns list |
| **Schema** | `df.schema` | Schema info | `df.schema` | Both return column-to-dtype mapping |
| **Column Types** | `df.column_types` | Storage info | N/A | JAXFrame tracks storage type (JAX/numpy/list) |
| **DataFrame Name** | `df.name` | No direct equivalent | N/A | JAXFrame-specific feature |
| **Naming** | `df.with_name(name)` | No direct equivalent | N/A | JAXFrame-specific feature |

## Data Access & Selection

| **JAXFrame Feature** | **JAXFrame Method/Property** | **Similar Polars Feature** | **Polars Method/Property** | **Notes** |
|----------------------|-------------------------------|----------------------------|----------------------------|-----------|
| **Column Access** | `df[column]` | Column selection | `df[column]` | Similar but JAXFrame returns copies |
| **Column Containment** | `df.__contains__(key)` | Column existence | `key in df.columns` | JAXFrame has built-in method |
| **Row Access** | `df.get_row(index)` | Row access | `df.row(index)` | Return dict vs tuple by default |
| **Column Selection** | `df.select_columns(columns)` | Column selection | `df.select(columns)` | JAXFrame takes list, Polars more flexible |
| **Data Export** | `df.to_dict(copy=True)` | To dictionary | `df.to_dict(as_series=False)` | Different default behaviors |
| **JAX Integration** | `df.to_jax_dict()` | No direct equivalent | `df.to_jax()` | Polars has to_jax but different approach |
| **NumPy Integration** | `df.to_numpy_dict()` | To NumPy | `df.to_numpy()` | Different output formats |
| **JAX Column Selection** | `df.get_jax_columns(columns)` | Column selection + conversion | `df.select(columns).to_jax()` | JAXFrame more direct |

## Data Manipulation

| **JAXFrame Feature** | **JAXFrame Method/Property** | **Similar Polars Feature** | **Polars Method/Property** | **Notes** |
|----------------------|-------------------------------|----------------------------|----------------------------|-----------|
| **Add Column** | `df.add_column(name, values)` | Add column | `df.with_columns(col=values)` | Different syntax |
| **Remove Column** | `df.remove_column(name)` | Drop column | `df.drop(name)` | Similar functionality |
| **Add Row** | `df.add_row(row_data)` | No direct equivalent | `df.vstack(new_row_df)` | Polars requires DataFrame for new rows |
| **Remove Row** | `df.remove_row(index)` | Filter by index | `df.filter(pl.int_range(len(df)) != index)` | Polars more expression-based |
| **Concatenation** | `df.concat(other, axis=0)` | Vertical concatenation | `df.vstack(other)` | Similar for axis=0 |
| **Concatenation** | `df.concat(other, axis=1)` | Horizontal concatenation | `df.hstack(other)` | Similar for axis=1 |
| **Static Concatenation** | `DataFrame.concat_dataframes(dfs)` | Concatenate multiple | `pl.concat(dfs)` | Static method vs function |

## Joins & Relationships

| **JAXFrame Feature** | **JAXFrame Method/Property** | **Similar Polars Feature** | **Polars Method/Property** | **Notes** |
|----------------------|-------------------------------|----------------------------|----------------------------|-----------|
| **Inner Join** | `df.join(other, on=cols, how='inner')` | Inner join | `df.join(other, on=cols, how='inner')` | Very similar interface |
| **Left Join** | `df.join(other, on=cols, how='left')` | Left join | `df.join(other, on=cols, how='left')` | Similar interface |
| **Join with Aliases** | `df.join(other, source=cols, target=cols)` | Join with aliases | `df.join(other, left_on=cols, right_on=cols)` | Different parameter names |
| **Lookup Tables** | `df.is_valid_lookup_table(id_cols)` | No direct equivalent | Custom validation logic | JAXFrame-specific feature |
| **Lookup Table Update** | `df.update_lookup_table(other, id_cols)` | Update/upsert | `df.update(other, on=id_cols)` | Similar concept, different implementation |
| **Lookup Table Replace** | `df.replace_lookup_table(other, id_cols)` | No direct equivalent | Custom logic needed | JAXFrame-specific feature |

## Display & Formatting

| **JAXFrame Feature** | **JAXFrame Method/Property** | **Similar Polars Feature** | **Polars Method/Property** | **Notes** |
|----------------------|-------------------------------|----------------------------|----------------------------|-----------|
| **String Representation** | `str(df)` | String representation | `str(df)` | Both use Unicode tables |
| **Pretty Print** | `df.to_string(max_rows, max_cols)` | No direct method | Built into `str(df)` | JAXFrame has Polars-like formatting |
| **Repr** | `repr(df)` | Repr | `repr(df)` | Similar output |
| **Environment Variables** | `POLARS_FMT_*` support | Environment config | `POLARS_FMT_*` variables | JAXFrame mimics Polars formatting |

## Data Transformation (JAXFrame-Specific)

| **JAXFrame Feature** | **JAXFrame Method** | **Similar Polars Feature** | **Polars Method** | **Notes** |
|----------------------|---------------------|----------------------------|-------------------|-----------|
| **Wide to Long** | `wide_to_long_masked(df, id_cols)` | Melt/Unpivot | `df.melt(id_vars=id_cols)` | JAXFrame handles masks |
| **Long to Wide** | `long_to_wide_masked(df, id_cols, value_col)` | Pivot | `df.pivot(index=id_cols, values=value_col)` | JAXFrame creates masks |
| **JAX Array Conversion** | `wide_df_to_jax_arrays(df, id_cols)` | To JAX | `df.to_jax()` | JAXFrame returns values+masks |
| **JAX to DataFrame** | `jax_arrays_to_wide_df(values, masks, ids)` | From JAX | `pl.DataFrame(jax_array)` | JAXFrame handles masks |
| **Roundtrip Conversion** | `roundtrip_wide_jax_conversion(df, id_cols)` | No equivalent | N/A | JAXFrame-specific testing utility |

## Comparison & Equality

| **JAXFrame Feature** | **JAXFrame Method/Property** | **Similar Polars Feature** | **Polars Method/Property** | **Notes** |
|----------------------|-------------------------------|----------------------------|----------------------------|-----------|
| **DataFrame Equality** | `df.__eq__(other)` | DataFrame equality | `df.equals(other)` | Different method names |
| **Value Comparison** | `df._values_equal(val1, val2)` | No direct equivalent | Element comparison in expressions | JAXFrame helper method |

## Integration & Export

| **JAXFrame Feature** | **JAXFrame Method** | **Similar Polars Feature** | **Polars Method** | **Notes** |
|----------------------|---------------------|----------------------------|-------------------|-----------|
| **Pandas Export** | `df.to_pandas()` | Pandas export | `df.to_pandas()` | Identical interface |
| **Copy/Clone** | Built into operations | Clone | `df.clone()` | JAXFrame immutable by design |

## Missing Polars Features in JAXFrame

JAXFrame does **not** have equivalents for many advanced Polars features:

### Query & Filtering
- `df.filter()` - Expression-based filtering
- `df.with_columns()` - Add/modify columns with expressions  
- `df.select()` - Expression-based column selection
- `df.group_by().agg()` - Group by operations
- `df.sort()` - Sorting operations
- `df.unique()` - Remove duplicates

### Advanced Operations
- `df.lazy()` - Lazy evaluation
- `df.collect()` - Execute lazy operations
- `df.explode()` - Explode list columns
- `df.pivot()` - Advanced pivot operations
- `df.window()` - Window functions
- `df.rolling()` - Rolling window operations

### Data Types & Casting
- `df.cast()` - Type casting
- `df.with_columns(pl.col().cast())` - Column-wise casting
- Complex data types (Lists, Structs, etc.)

### I/O Operations
- `pl.read_csv()`, `pl.read_parquet()`, etc.
- `df.write_csv()`, `df.write_parquet()`, etc.
- Streaming I/O operations

### String Operations
- `pl.col().str.*` - String manipulation
- Regular expressions
- String parsing

### Date/Time Operations
- `pl.col().dt.*` - DateTime operations
- Time zone handling
- Date parsing and formatting

## Unique JAXFrame Features Not in Polars

1. **JAX-First Design**: Optimized for JAX computational graphs
2. **Mask-Aware Transformations**: Built-in support for masked data
3. **Immutable by Design**: All operations return new DataFrames
4. **Mixed Storage Types**: Tracks whether columns are lists, NumPy arrays, or JAX arrays
5. **Fast Constructors**: Type-specific constructors for performance
6. **Named DataFrames**: DataFrames can have names
7. **Lookup Table Operations**: Specialized methods for lookup table management

## Priority Alignment Candidates

Based on common usage patterns and API consistency, these JAXFrame features could be aligned with Polars:

### High Priority (Easy Wins)
1. **Column Selection**: `df.select_columns()` → `df.select()`
2. **Drop Columns**: `df.remove_column()` → `df.drop()`
3. **DataFrame Equality**: `df.__eq__()` → `df.equals()`
4. **Data Types**: `df.dtypes()` → `df.dtypes` (property)
5. **Join Parameters**: `source/target` → `left_on/right_on`

### Medium Priority (API Extensions)
6. **Add Columns**: `df.add_column()` → `df.with_columns()`
7. **Concatenation**: Add `df.vstack()` and `df.hstack()` aliases
8. **Clone Method**: Add `df.clone()` method
9. **Head/Tail**: Add `df.head()` and `df.tail()` methods
10. **Sample**: Add `df.sample()` method

### Low Priority (Complex Features)
11. **Basic Filtering**: Add simple `df.filter()` method
12. **Basic Sorting**: Add simple `df.sort()` method
13. **Unique Rows**: Add `df.unique()` method
14. **Column Renaming**: Add `df.rename()` method
15. **Basic Aggregations**: Add `df.sum()`, `df.mean()`, etc.

## Summary

**JAXFrame ≈ 15-20% of Polars' feature set**, but with specialized focus on JAX workflows and masked data operations that Polars doesn't directly support.

JAXFrame is a **lightweight, specialized DataFrame library** focused on:
- JAX integration and computational graph preservation
- Masked data handling for scientific computing
- Immutable data structures
- Mixed storage type support

Polars is a **comprehensive, production-ready DataFrame library** with:
- Full SQL-like query capabilities
- Lazy evaluation and query optimization
- Extensive I/O support
- Advanced data types and operations
- High-performance Rust implementation

The alignment opportunities focus on making JAXFrame's API more familiar to Polars users while preserving its JAX-focused specialization.