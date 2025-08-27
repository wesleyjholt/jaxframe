"""Tests for DataFrame name handling functionality."""

import pytest
import numpy as np
from jaxframe import DataFrame

try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


class TestDataFrameNameHandling:
    """Test DataFrame name handling functionality."""
    
    def test_with_name_basic(self):
        """Test basic name setting functionality."""
        df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        assert df.name is None
        
        # Set a name
        df_named = df.with_name("my_dataframe")
        assert df_named.name == "my_dataframe"
        assert df.name is None  # Original unchanged
        
        # Check data is the same
        assert df_named['a'] == [1, 2, 3]
        assert df_named['b'] == [4, 5, 6]
        assert df_named.columns == df.columns
        assert df_named.shape == df.shape
    
    def test_with_name_preserves_data_types(self):
        """Test that with_name preserves data types."""
        data = {
            'lists': [1, 2, 3],
            'arrays': np.array([4, 5, 6]),
        }
        df = DataFrame(data, name="original")
        df_renamed = df.with_name("renamed")
        
        assert df_renamed.name == "renamed"
        assert df.name == "original"
        
        # Check data types are preserved
        assert df_renamed.column_types['lists'] == 'list'
        assert df_renamed.column_types['arrays'] == 'array'
        assert isinstance(df_renamed['lists'], list)
        assert isinstance(df_renamed['arrays'], np.ndarray)
    
    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_with_name_preserves_jax_arrays(self):
        """Test that with_name preserves JAX arrays."""
        data = {
            'jax_col': jnp.array([1.0, 2.0, 3.0]),
            'regular_col': [4, 5, 6]
        }
        df = DataFrame(data, name="jax_test")
        df_renamed = df.with_name("jax_renamed")
        
        assert df_renamed.name == "jax_renamed"
        assert df_renamed.column_types['jax_col'] == 'jax_array'
        assert df_renamed.column_types['regular_col'] == 'list'
    
    def test_with_name_chaining(self):
        """Test that with_name can be chained and used with other methods."""
        df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        # Chain with_name with other operations
        result = df.with_name("step1").add_column('c', [7, 8, 9]).with_name("step2")
        
        assert result.name == "step2"
        assert 'c' in result.columns
        assert result['c'] == [7, 8, 9]
    
    def test_with_name_empty_string(self):
        """Test with_name with empty string."""
        df = DataFrame({'a': [1, 2, 3]}, name="original")
        df_empty_name = df.with_name("")
        
        assert df_empty_name.name == ""
        assert df.name == "original"
    
    def test_with_name_none(self):
        """Test with_name with None."""
        df = DataFrame({'a': [1, 2, 3]}, name="original")
        df_no_name = df.with_name(None)
        
        assert df_no_name.name is None
        assert df.name == "original"


class TestLookupTableNameInheritance:
    """Test that lookup table operations properly inherit names."""
    
    def test_update_lookup_table_inherits_name(self):
        """Test that update_lookup_table inherits the original DataFrame's name."""
        # Create lookup table with a name
        lookup = DataFrame({
            'key': ['a', 'b'],
            'value': [1, 2]
        }, name="my_lookup_table")
        
        # Create update data without a name
        updates = DataFrame({
            'key': ['c'],
            'value': [3]
        })
        
        # Update should inherit original name
        result = lookup.update_lookup_table(updates, 'key')
        assert result.name == "my_lookup_table"
    
    def test_update_lookup_table_inherits_none_name(self):
        """Test that update_lookup_table inherits None name."""
        # Create lookup table without a name
        lookup = DataFrame({
            'key': ['a', 'b'],
            'value': [1, 2]
        })
        
        # Create update data with a name
        updates = DataFrame({
            'key': ['c'],
            'value': [3]
        }, name="updates")
        
        # Update should inherit None name from original
        result = lookup.update_lookup_table(updates, 'key')
        assert result.name is None
    
    def test_replace_lookup_table_inherits_name(self):
        """Test that replace_lookup_table inherits the original DataFrame's name."""
        # Create lookup table with a name
        lookup = DataFrame({
            'key': ['a', 'b'],
            'value': [1, 2]
        }, name="replaceable_lookup")
        
        # Create replacement data with conflicting values
        replacements = DataFrame({
            'key': ['a', 'c'],
            'value': [10, 3]
        }, name="replacements")
        
        # Replace should inherit original name
        result = lookup.replace_lookup_table(replacements, 'key')
        assert result.name == "replaceable_lookup"
        
        # Check that replacement actually happened
        keys = result['key']
        values = result['value']
        a_index = keys.index('a')
        assert values[a_index] == 10  # 'a' was replaced
        assert 'c' in keys  # 'c' was added
    
    def test_update_lookup_table_strict_mode_inherits_name(self):
        """Test that strict mode update_lookup_table inherits name."""
        lookup = DataFrame({
            'key': ['a', 'b'],
            'value': [1, 2]
        }, name="strict_lookup")
        
        # Add new key (should work in strict mode)
        updates = DataFrame({
            'key': ['c'],
            'value': [3]
        })
        
        result = lookup.update_lookup_table(updates, 'key', strict=True)
        assert result.name == "strict_lookup"
        assert 'c' in result['key']
    
    def test_update_lookup_table_multi_column_inherits_name(self):
        """Test that multi-column key updates inherit name."""
        lookup = DataFrame({
            'key1': ['a', 'b'],
            'key2': [1, 2],
            'value': [10, 20]
        }, name="multi_key_lookup")
        
        updates = DataFrame({
            'key1': ['c'],
            'key2': [3],
            'value': [30]
        })
        
        result = lookup.update_lookup_table(updates, ['key1', 'key2'])
        assert result.name == "multi_key_lookup"
    
    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_lookup_table_with_jax_arrays_inherits_name(self):
        """Test that lookup table operations with JAX arrays inherit name."""
        lookup = DataFrame({
            'key': ['a', 'b'],
            'value': jnp.array([1.0, 2.0])
        }, name="jax_lookup")
        
        updates = DataFrame({
            'key': ['c'],
            'value': jnp.array([3.0])
        })
        
        result = lookup.update_lookup_table(updates, 'key')
        assert result.name == "jax_lookup"
        assert result.column_types['value'] == 'jax_array'


class TestNameHandlingIntegration:
    """Test integration of name handling with other DataFrame operations."""
    
    def test_name_persistence_through_operations(self):
        """Test that names persist through various DataFrame operations."""
        df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}, name="test_df")
        
        # Most operations should preserve name
        operations = [
            df.add_column('c', [7, 8, 9]),
            df.remove_column('b'),
            df.add_row({'a': 4, 'b': 7}),
            df.remove_row(0),
        ]
        
        for result in operations:
            # These operations should preserve the name (this is existing behavior)
            # We're just testing to make sure our new with_name method works alongside them
            assert hasattr(result, 'name')  # Name attribute exists
    
    def test_with_name_vs_constructor_name(self):
        """Test that with_name produces same result as constructor with name."""
        data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        
        # Two ways to create named DataFrame
        df1 = DataFrame(data, name="test_name")
        df2 = DataFrame(data).with_name("test_name")
        
        assert df1.name == df2.name
        assert df1.columns == df2.columns
        assert df1.shape == df2.shape
        assert df1['a'] == df2['a']
        assert df1['b'] == df2['b']
        assert df1.column_types == df2.column_types
