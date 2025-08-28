"""Tests for DataFrame join functionality with SQL-style join types."""

import pytest
import numpy as np
from jaxframe import DataFrame

try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


class TestDataFrameJoins:
    """Test DataFrame join functionality with different join types."""
    
    def setup_method(self):
        """Set up test data for joins."""
        # Left DataFrame (customers)
        self.customers = DataFrame({
            'customer_id': ['A', 'B', 'C', 'D'],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
            'age': [25, 30, 35, 28]
        }, name="customers")
        
        # Right DataFrame (orders) - not all customers have orders
        self.orders = DataFrame({
            'customer_id': ['A', 'B', 'E'],  # Note: C, D missing; E is new
            'order_amount': [100.0, 200.0, 150.0],
            'order_date': ['2023-01-01', '2023-01-02', '2023-01-03']
        }, name="orders")
        
        # For testing multi-column joins
        self.products = DataFrame({
            'customer_id': ['A', 'B', 'E'],
            'product': ['Widget', 'Gadget', 'Tool'],
            'price': [10.0, 20.0, 15.0]
        }, name="products")
    
    def test_left_join_basic(self):
        """Test basic left join functionality."""
        result = self.customers.join(
            self.orders, 
            on='customer_id',
            source='order_amount',
            how='left'
        )
        
        # Should have all customers (4 rows)
        assert len(result) == 4
        assert set(result['customer_id']) == {'A', 'B', 'C', 'D'}
        
        # Check specific values
        customer_ids = result['customer_id']
        order_amounts = result['orders/order_amount']
        
        # A and B should have order amounts
        a_idx = customer_ids.index('A')
        b_idx = customer_ids.index('B')
        assert order_amounts[a_idx] == 100.0
        assert order_amounts[b_idx] == 200.0
        
        # C and D should have NaN
        c_idx = customer_ids.index('C')
        d_idx = customer_ids.index('D')
        assert np.isnan(order_amounts[c_idx])
        assert np.isnan(order_amounts[d_idx])
    
    def test_inner_join_basic(self):
        """Test basic inner join functionality."""
        result = self.customers.join(
            self.orders,
            on='customer_id', 
            source='order_amount',
            how='inner'
        )
        
        # Should only have customers with orders (2 rows)
        assert len(result) == 2
        assert set(result['customer_id']) == {'A', 'B'}
        
        # No NaN values should be present
        order_amounts = result['orders/order_amount']
        assert not any(np.isnan(val) for val in order_amounts)
    
    def test_right_join_basic(self):
        """Test basic right join functionality."""
        result = self.customers.join(
            self.orders,
            on='customer_id',
            source='order_amount', 
            how='right'
        )
        
        # Should have all orders (3 rows) 
        assert len(result) == 3
        assert set(result['customer_id']) == {'A', 'B', 'E'}
        
        # E should have NaN for customer info but real order amount
        customer_ids = result['customer_id']
        names = result['name']
        order_amounts = result['orders/order_amount']
        
        e_idx = customer_ids.index('E')
        assert np.isnan(names[e_idx])  # No customer name for E
        assert order_amounts[e_idx] == 150.0  # But has order amount
    
    def test_outer_join_basic(self):
        """Test basic outer join functionality.""" 
        result = self.customers.join(
            self.orders,
            on='customer_id',
            source='order_amount',
            how='outer'
        )
        
        # Should have all unique customer_ids (5 rows: A, B, C, D, E)
        assert len(result) == 5
        assert set(result['customer_id']) == {'A', 'B', 'C', 'D', 'E'}
        
        # Check for appropriate NaN values
        customer_ids = result['customer_id']
        names = result['name']
        order_amounts = result['orders/order_amount']
        
        # C and D should have names but no order amounts
        c_idx = customer_ids.index('C')
        d_idx = customer_ids.index('D')
        assert names[c_idx] == 'Charlie'
        assert names[d_idx] == 'Diana'
        assert np.isnan(order_amounts[c_idx])
        assert np.isnan(order_amounts[d_idx])
        
        # E should have order amount but no name
        e_idx = customer_ids.index('E')
        assert np.isnan(names[e_idx])
        assert order_amounts[e_idx] == 150.0
    
    def test_multi_column_joins(self):
        """Test joins with multiple columns."""
        result = self.customers.join(
            self.orders,
            on='customer_id',
            source=['order_amount', 'order_date'],
            how='left'
        )
        
        assert len(result) == 4
        assert 'orders/order_amount' in result.columns
        assert 'orders/order_date' in result.columns
        
        # Check that both columns have NaN for missing customers
        customer_ids = result['customer_id']
        order_amounts = result['orders/order_amount']
        order_dates = result['orders/order_date']
        
        c_idx = customer_ids.index('C')
        assert np.isnan(order_amounts[c_idx])
        assert np.isnan(order_dates[c_idx])
    
    def test_custom_target_column_names(self):
        """Test joins with custom target column names."""
        result = self.customers.join(
            self.orders,
            on='customer_id',
            source=['order_amount', 'order_date'],
            target=['amount', 'date'],
            how='left'
        )
        
        assert 'amount' in result.columns
        assert 'date' in result.columns
        assert 'orders/order_amount' not in result.columns
        assert 'orders/order_date' not in result.columns
    
    def test_join_preserves_data_types(self):
        """Test that joins preserve original data types."""
        # Create test data with different types
        left_df = DataFrame({
            'key': ['A', 'B'],
            'list_col': [1, 2],  # List type
            'array_col': np.array([10, 20])  # Array type
        })
        
        right_df = DataFrame({
            'key': ['A', 'B'], 
            'value_list': [100, 200],  # List type
            'value_array': np.array([1000.0, 2000.0])  # Array type
        })
        
        result = left_df.join(right_df, 'key', ['value_list', 'value_array'], how='left')
        
        # Check that types are preserved
        assert result.column_types['list_col'] == 'list'
        assert result.column_types['array_col'] == 'array'
        assert result.column_types['value_list'] == 'list'
        assert result.column_types['value_array'] == 'array'
    
    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_join_with_jax_arrays(self):
        """Test joins with JAX arrays."""
        left_df = DataFrame({
            'key': ['A', 'B', 'C'],
            'jax_col': jnp.array([1.0, 2.0, 3.0])
        })
        
        right_df = DataFrame({
            'key': ['A', 'B'],
            'jax_value': jnp.array([10.0, 20.0])
        })
        
        result = left_df.join(right_df, 'key', 'jax_value', how='left')
        
        # Check that JAX types are preserved
        assert result.column_types['jax_col'] == 'jax_array'
        assert result.column_types['jax_value'] == 'jax_array'
        
        # Check that NaN is properly handled for JAX arrays
        jax_values = result['jax_value']
        assert not np.isnan(jax_values[0])  # A has value
        assert not np.isnan(jax_values[1])  # B has value
        assert np.isnan(jax_values[2])  # C has NaN
    
    def test_invalid_join_type(self):
        """Test that invalid join types raise appropriate errors."""
        with pytest.raises(ValueError, match="Invalid join type 'invalid'"):
            self.customers.join(
                self.orders,
                on='customer_id',
                source='order_amount',
                how='invalid'
            )
    
    def test_duplicate_keys_in_right_dataframe(self):
        """Test that duplicate keys in right DataFrame raise error."""
        duplicate_orders = DataFrame({
            'customer_id': ['A', 'A', 'B'],  # Duplicate A
            'order_amount': [100.0, 150.0, 200.0]
        })
        
        with pytest.raises(ValueError, match="Duplicate values found in 'customer_id'"):
            self.customers.join(
                duplicate_orders,
                on='customer_id',
                source='order_amount',
                how='left'
            )
    
    def test_missing_columns(self):
        """Test appropriate errors for missing columns."""
        # Missing on_column in left DataFrame
        with pytest.raises(KeyError, match="Column 'missing' not found in this DataFrame"):
            self.customers.join(
                self.orders,
                on='missing',
                source='order_amount'
            )
        
        # Missing on_column in right DataFrame - create a DataFrame that doesn't have the column
        orders_missing_col = DataFrame({
            'different_id': ['A', 'B', 'E'],
            'order_amount': [100.0, 200.0, 150.0]
        })
        
        with pytest.raises(KeyError, match="Column 'customer_id' not found in other DataFrame"):
            self.customers.join(
                orders_missing_col,
                on='customer_id',
                source='order_amount'
            )
        
        # Missing source_column in right DataFrame
        with pytest.raises(KeyError, match="Column 'missing' not found in other DataFrame"):
            self.customers.join(
                self.orders,
                on='customer_id',
                source='missing'
            )
    
    def test_target_column_length_mismatch(self):
        """Test error when target length doesn't match source length."""
        with pytest.raises(ValueError, match="Length of target list"):
            self.customers.join(
                self.orders,
                on='customer_id',
                source=['order_amount', 'order_date'],
                target=['amount_only']  # Only one target for two sources
            )
    
    def test_backward_compatibility_default_inner_join(self):
        """Test that default behavior is inner join."""
        # Call without specifying 'how' - should default to inner join
        result_default = self.customers.join(
            self.orders,
            on='customer_id',
            source='order_amount'
        )
        
        result_explicit = self.customers.join(
            self.orders,
            on='customer_id', 
            source='order_amount',
            how='inner'
        )
        
        # Results should be identical
        assert len(result_default) == len(result_explicit)
        assert result_default.columns == result_explicit.columns
        
        # Check that all values match
        for col in result_default.columns:
            default_vals = result_default[col]
            explicit_vals = result_explicit[col]
            for i in range(len(default_vals)):
                default_val = default_vals[i]
                explicit_val = explicit_vals[i]
                if isinstance(default_val, float) and np.isnan(default_val):
                    assert np.isnan(explicit_val)
                else:
                    assert default_val == explicit_val


class TestJoinEdgeCases:
    """Test edge cases for join functionality."""
    
    def test_join_with_empty_dataframe(self):
        """Test joins with empty DataFrames."""
        # Note: Can't create truly empty DataFrame, so create minimal one
        left_df = DataFrame({'key': ['A'], 'value': [1]})
        right_df = DataFrame({'key': ['B'], 'value': [2]})  # No matching keys
        
        # Inner join should result in empty-like result
        result = left_df.join(right_df, 'key', 'value', how='inner')
        assert len(result) == 0
    
    def test_join_with_single_row_dataframes(self):
        """Test joins with single-row DataFrames.""" 
        left_df = DataFrame({'key': ['A'], 'name': ['Alice']})
        right_df = DataFrame({'key': ['A'], 'score': [95.0]})
        
        result = left_df.join(right_df, 'key', 'score', how='inner')
        assert len(result) == 1
        assert result['name'][0] == 'Alice'
        assert result['score'][0] == 95.0
    
    def test_join_maintains_order(self):
        """Test that join maintains order based on join type."""
        left_df = DataFrame({
            'key': ['C', 'A', 'B'],  # Intentionally not sorted
            'value': [3, 1, 2]
        })
        
        right_df = DataFrame({
            'key': ['A', 'B', 'C'],  # Different order
            'score': [10, 20, 30]
        })
        
        # Left join should preserve left DataFrame order
        result_left = left_df.join(right_df, 'key', 'score', how='left')
        assert result_left['key'] == ['C', 'A', 'B']
        
        # Inner join should preserve left DataFrame order for matching keys
        result_inner = left_df.join(right_df, 'key', 'score', how='inner')
        assert result_inner['key'] == ['C', 'A', 'B']  # Same order as left for matching keys
        
        # Right join should preserve right DataFrame order
        result_right = left_df.join(right_df, 'key', 'score', how='right')
        assert result_right['key'] == ['A', 'B', 'C']  # Same order as right
