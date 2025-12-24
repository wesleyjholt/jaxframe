"""
Comprehensive tests for categorical column tracking in DataFrame.
"""
import jax.numpy as jnp
import numpy as np
from src.jaxframe import DataFrame


def test_automatic_categorical_inference():
    """Test that categorical flags are inferred correctly by default."""
    print("\n" + "="*70)
    print("TEST: Automatic Categorical Inference")
    print("="*70)
    
    df = DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],           # String - should be categorical
        'age': [25, 30, 35],                            # Int - should be categorical by default
        'salary': [50000.0, 60000.0, 70000.0],        # Float - should be non-categorical
        'active': [True, False, True],                 # Bool - should be categorical by default
        'scores': jnp.array([1.5, 2.5, 3.5])          # JAX float - should be non-categorical
    })
    
    print(f"\nDataFrame:\n{df}")
    print(f"\nCategorical status: {df.categorical}")
    
    assert df.is_categorical('name') == True, "String columns should be categorical"
    assert df.is_categorical('age') == True, "Int columns should be categorical by default"
    assert df.is_categorical('salary') == False, "Float columns should be non-categorical"
    assert df.is_categorical('active') == True, "Bool columns should be categorical by default"
    assert df.is_categorical('scores') == False, "JAX float columns should be non-categorical"
    
    print("\n✓ All automatic inference tests passed!")


def test_get_categorical_columns():
    """Test getting lists of categorical and non-categorical columns."""
    print("\n" + "="*70)
    print("TEST: Get Categorical/Non-Categorical Columns")
    print("="*70)
    
    df = DataFrame({
        'id': [1, 2, 3],
        'name': ['A', 'B', 'C'],
        'value': [1.1, 2.2, 3.3]
    })
    
    cat_cols = df.get_categorical_columns()
    non_cat_cols = df.get_non_categorical_columns()
    
    print(f"\nCategorical columns: {cat_cols}")
    print(f"Non-categorical columns: {non_cat_cols}")
    
    assert 'id' in cat_cols
    assert 'name' in cat_cols
    assert 'value' in non_cat_cols
    
    print("\n✓ Get categorical columns tests passed!")


def test_int_column_can_be_non_categorical():
    """Test that int columns can be forced to non-categorical."""
    print("\n" + "="*70)
    print("TEST: Int Column Can Be Non-Categorical")
    print("="*70)
    
    df = DataFrame({
        'count': [10, 20, 30],
        'name': ['A', 'B', 'C']
    })
    
    print(f"\nOriginal categorical status: {df.categorical}")
    assert df.is_categorical('count') == True, "Int should be categorical by default"
    
    # Force int column to be non-categorical
    df2 = df.as_non_categorical('count')
    print(f"After as_non_categorical('count'): {df2.categorical}")
    
    assert df2.is_categorical('count') == False
    assert df2.is_categorical('name') == True  # name should remain categorical
    
    print("\n✓ Int column can be marked as non-categorical!")


def test_int_column_stays_categorical():
    """Test that int columns can stay categorical."""
    print("\n" + "="*70)
    print("TEST: Int Column Can Stay Categorical")
    print("="*70)
    
    df = DataFrame({
        'category_id': [1, 2, 3],
        'value': [100.0, 200.0, 300.0]
    })
    
    # Force it to be categorical (even though it already is)
    df2 = df.as_categorical('category_id')
    
    assert df2.is_categorical('category_id') == True
    print("\n✓ Int column can remain categorical!")


def test_float_cannot_be_categorical():
    """Test that float columns cannot be forced to categorical."""
    print("\n" + "="*70)
    print("TEST: Float Column Cannot Be Categorical")
    print("="*70)
    
    df = DataFrame({
        'price': [10.5, 20.5, 30.5],
        'name': ['A', 'B', 'C']
    })
    
    print(f"\nOriginal categorical status: {df.categorical}")
    
    try:
        df2 = df.as_categorical('price')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"\n✓ Correctly raised error: {e}")
        assert "float-like" in str(e).lower()
        assert "cannot" in str(e).lower()


def test_string_cannot_be_non_categorical():
    """Test that string columns cannot be forced to non-categorical."""
    print("\n" + "="*70)
    print("TEST: String Column Cannot Be Non-Categorical")
    print("="*70)
    
    df = DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'value': [1.0, 2.0, 3.0]
    })
    
    print(f"\nOriginal categorical status: {df.categorical}")
    
    try:
        df2 = df.as_non_categorical('name')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"\n✓ Correctly raised error: {e}")
        assert "string-like" in str(e).lower()
        assert "cannot" in str(e).lower()


def test_explicit_categorical_in_constructor():
    """Test providing explicit categorical flags in constructor."""
    print("\n" + "="*70)
    print("TEST: Explicit Categorical in Constructor")
    print("="*70)
    
    # Make int column non-categorical from the start
    df = DataFrame(
        {
            'id': [1, 2, 3],
            'count': [10, 20, 30],
            'name': ['A', 'B', 'C']
        },
        categorical={'id': False, 'count': True}  # Explicit override for int columns
    )
    
    print(f"\nCategorical status: {df.categorical}")
    
    assert df.is_categorical('id') == False, "id should be non-categorical (explicit)"
    assert df.is_categorical('count') == True, "count should be categorical (explicit)"
    assert df.is_categorical('name') == True, "name should be categorical (default for string)"
    
    print("\n✓ Explicit categorical flags work in constructor!")


def test_constructor_rejects_invalid_categorical():
    """Test that constructor rejects invalid categorical specifications."""
    print("\n" + "="*70)
    print("TEST: Constructor Rejects Invalid Categorical")
    print("="*70)
    
    # Try to make float categorical
    try:
        df = DataFrame(
            {'value': [1.0, 2.0, 3.0]},
            categorical={'value': True}
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"\n✓ Correctly rejected float as categorical: {e}")
    
    # Try to make string non-categorical
    try:
        df = DataFrame(
            {'name': ['A', 'B', 'C']},
            categorical={'name': False}
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"\n✓ Correctly rejected string as non-categorical: {e}")


def test_multiple_columns_as_categorical():
    """Test marking multiple columns as categorical at once."""
    print("\n" + "="*70)
    print("TEST: Multiple Columns As Categorical")
    print("="*70)
    
    df = DataFrame({
        'id': [1, 2, 3],
        'category': [10, 20, 30],
        'active': [True, False, True]
    })
    
    # Make all non-categorical
    df = df.as_non_categorical(['id', 'category', 'active'])
    
    print(f"\nAfter as_non_categorical: {df.categorical}")
    assert all(not df.is_categorical(col) for col in ['id', 'category', 'active'])
    
    # Make all categorical again
    df = df.as_categorical(['id', 'category', 'active'])
    
    print(f"After as_categorical: {df.categorical}")
    assert all(df.is_categorical(col) for col in ['id', 'category', 'active'])
    
    print("\n✓ Multiple column operations work!")


def test_numpy_arrays():
    """Test categorical inference with numpy arrays."""
    print("\n" + "="*70)
    print("TEST: NumPy Arrays")
    print("="*70)
    
    df = DataFrame({
        'int_arr': np.array([1, 2, 3]),
        'float_arr': np.array([1.0, 2.0, 3.0]),
        'bool_arr': np.array([True, False, True])
    })
    
    print(f"\nCategorical status: {df.categorical}")
    
    assert df.is_categorical('int_arr') == True
    assert df.is_categorical('float_arr') == False
    assert df.is_categorical('bool_arr') == True
    
    print("\n✓ NumPy array categorical inference works!")


def test_jax_arrays():
    """Test categorical inference with JAX arrays."""
    print("\n" + "="*70)
    print("TEST: JAX Arrays")
    print("="*70)
    
    df = DataFrame({
        'int_jax': jnp.array([1, 2, 3]),
        'float_jax': jnp.array([1.0, 2.0, 3.0])
    })
    
    print(f"\nCategorical status: {df.categorical}")
    
    assert df.is_categorical('int_jax') == True
    assert df.is_categorical('float_jax') == False
    
    print("\n✓ JAX array categorical inference works!")


def test_immutability():
    """Test that categorical modifications return new DataFrames."""
    print("\n" + "="*70)
    print("TEST: Immutability")
    print("="*70)
    
    df1 = DataFrame({
        'id': [1, 2, 3],
        'value': [10.0, 20.0, 30.0]
    })
    
    original_cat = df1.is_categorical('id')
    print(f"\nOriginal df1.is_categorical('id'): {original_cat}")
    
    df2 = df1.as_non_categorical('id')
    
    print(f"After creating df2, df1.is_categorical('id'): {df1.is_categorical('id')}")
    print(f"df2.is_categorical('id'): {df2.is_categorical('id')}")
    
    assert df1.is_categorical('id') == original_cat, "Original DataFrame should be unchanged"
    assert df2.is_categorical('id') != original_cat, "New DataFrame should be different"
    
    print("\n✓ Immutability preserved!")


def test_error_messages():
    """Test that error messages are clear and helpful."""
    print("\n" + "="*70)
    print("TEST: Error Messages")
    print("="*70)
    
    df = DataFrame({
        'price': [10.0, 20.0, 30.0],
        'name': ['A', 'B', 'C']
    })
    
    # Test float to categorical error
    try:
        df.as_categorical('price')
        assert False
    except ValueError as e:
        print(f"\nFloat to categorical error: {e}")
        assert 'price' in str(e)
        assert 'float' in str(e).lower()
        assert 'cannot' in str(e).lower()
    
    # Test string to non-categorical error
    try:
        df.as_non_categorical('name')
        assert False
    except ValueError as e:
        print(f"\nString to non-categorical error: {e}")
        assert 'name' in str(e)
        assert 'string' in str(e).lower()
        assert 'cannot' in str(e).lower()
    
    # Test non-existent column error
    try:
        df.is_categorical('nonexistent')
        assert False
    except KeyError as e:
        print(f"\nNon-existent column error: {e}")
        assert 'nonexistent' in str(e)
    
    print("\n✓ Error messages are clear and helpful!")


if __name__ == '__main__':
    test_automatic_categorical_inference()
    test_get_categorical_columns()
    test_int_column_can_be_non_categorical()
    test_int_column_stays_categorical()
    test_float_cannot_be_categorical()
    test_string_cannot_be_non_categorical()
    test_explicit_categorical_in_constructor()
    test_constructor_rejects_invalid_categorical()
    test_multiple_columns_as_categorical()
    test_numpy_arrays()
    test_jax_arrays()
    test_immutability()
    test_error_messages()
    
    print("\n" + "="*70)
    print("ALL CATEGORICAL TESTS PASSED! ✓")
    print("="*70)
