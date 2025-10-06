# JAXFrame Examples

This directory contains practical examples demonstrating JAXFrame's features and capabilities. Each example focuses on specific aspects of the library and includes performance considerations.

## Examples Overview

### 01_basic_operations.py
**Basic DataFrame Operations**
- Creating DataFrames with mixed data types (JAX arrays, NumPy arrays, Python lists)
- Column access and type preservation
- Immutable operations and method chaining  
- Dictionary conversion and data inspection

**Key takeaways:**
- JAXFrame preserves the original data types of columns
- All operations return new DataFrames (immutability)
- Mixed data types are handled transparently

### 02_jax_integration.py
**JAX Integration - JIT and Gradients**  
- JAX arrays in DataFrames
- JIT compilation with DataFrames
- Gradient computation through DataFrame operations
- Tracer handling during compilation
- Real optimization example with gradient descent

**Key takeaways:**
- DataFrame operations preserve JAX computational graphs
- Gradients work correctly through DataFrame transformations
- JAX tracers are handled safely without ConcretizationTypeError
- DataFrames can be used in JIT-compiled functions

### 03_wide_to_long.py
**Wide-to-Long Data Transformations**
- Time series data with missing values
- Mask-based filtering during transformation
- Multiple variable handling with custom patterns
- Long-to-wide conversions with fill values
- Missing data analysis

**Key takeaways:**
- Missing data is handled automatically using mask columns
- Transformations preserve JAX array types
- Multiple variables can be transformed separately or together
- Custom regex patterns allow flexible column naming

### 04_masked_arrays.py
**MaskedArray Scientific Workflows**
- Converting wide DataFrames to MaskedArrays
- Statistical computations with missing data
- Group-wise analysis with masks
- Roundtrip conversions preserving computational graphs
- Scientific data patterns

**Key takeaways:**
- MaskedArrays combine JAX data + NumPy masks + metadata efficiently
- Statistical operations naturally handle missing values
- Roundtrip conversions preserve differentiability
- Useful for experimental data with missing observations

### 05_performance_analysis.py
**Performance Analysis and Optimization**
- DataFrame creation overhead measurement
- JIT compilation time comparison
- Memory usage analysis
- Performance optimization strategies
- Timing comparisons with pure JAX

**Key findings:**
- DataFrames add ~50% compilation overhead but minimal runtime overhead
- Memory usage is comparable to raw arrays
- Creating DataFrames inside tight loops can be 70x slower
- Best practices: use DataFrames for data prep, pure JAX for computation

### 06_polars_style_improvements.py
**Polars-Style API Design Ideas**
- Expression-based operations
- Lazy evaluation concepts
- Better filtering and grouping APIs
- JAX integration improvements
- Memory efficiency strategies

**Key concepts:**
- Separation of computational and metadata layers
- Copy-on-write semantics for better memory usage
- Lazy evaluation for query optimization
- JIT-friendly operation design

## Performance Summary

Based on the examples, here are the key performance characteristics:

### ✅ **Good Performance**
- Runtime operations: minimal overhead once compiled
- Memory usage: comparable to raw JAX arrays
- Gradient computation: works seamlessly
- Type preservation: efficient without conversions

### ⚠️ **Performance Considerations**
- JIT compilation: 50%+ longer compilation time
- DataFrame creation: avoid in tight loops
- Complex transformations: 10-50ms overhead for wide-to-long operations
- String operations: regex parsing during compilation

### 🎯 **Best Practices**
1. **Data Preparation**: Use DataFrames for data loading, cleaning, and initial transformations
2. **Computation**: Extract JAX arrays for intensive numerical computation
3. **JIT Functions**: Minimize DataFrame creation inside JIT-compiled functions
4. **Memory**: Prefer reusing DataFrames over creating many small ones

## Running the Examples

```bash
cd /path/to/jaxframe
python examples/01_basic_operations.py
python examples/02_jax_integration.py
python examples/03_wide_to_long.py
python examples/04_masked_arrays.py
python examples/05_performance_analysis.py
python examples/06_polars_style_improvements.py
```

## Requirements

- JAX (for array operations and JIT compilation)
- NumPy (for mask arrays and compatibility)
- Python 3.8+ (for type hints and modern features)

## Future Improvements

Based on these examples, potential improvements for a rewritten version include:

1. **API Design**: More Polars-like expression system
2. **Performance**: Separate computational from metadata layers
3. **Memory**: Arrow-style columnar storage with copy-on-write
4. **JAX Integration**: Minimize compilation overhead
5. **Lazy Evaluation**: Build query plans before execution

These examples demonstrate that JAXFrame successfully bridges DataFrame-style operations with JAX's functional programming model, though there are clear opportunities for optimization in a future rewrite.