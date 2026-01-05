"""
Tests for profiling instrumentation in jaxframe transforms.

This module tests that the profiling infrastructure works correctly and
doesn't break existing functionality.
"""

import os
import sys
from io import StringIO
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from jaxframe import DataFrame
from jaxframe.transform import (
    pivot_sparse,
    unpivot_sparse,
    to_masked_array,
    from_masked_array,
    _jaxframe_profile_enabled,
    _jaxframe_profile_limit,
    _should_profile,
    _profile_call_counters,
)

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@pytest.fixture(autouse=True)
def reset_profiling_state():
    """Reset profiling state before each test."""
    # Clear the call counters
    _profile_call_counters.clear()
    # Save original env vars
    orig_profile = os.environ.get("JAXFRAME_PROFILE_TRANSFORM")
    orig_limit = os.environ.get("JAXFRAME_PROFILE_LIMIT")
    
    yield
    
    # Restore original env vars
    if orig_profile is not None:
        os.environ["JAXFRAME_PROFILE_TRANSFORM"] = orig_profile
    elif "JAXFRAME_PROFILE_TRANSFORM" in os.environ:
        del os.environ["JAXFRAME_PROFILE_TRANSFORM"]
    
    if orig_limit is not None:
        os.environ["JAXFRAME_PROFILE_LIMIT"] = orig_limit
    elif "JAXFRAME_PROFILE_LIMIT" in os.environ:
        del os.environ["JAXFRAME_PROFILE_LIMIT"]
    
    # Clear counters again
    _profile_call_counters.clear()


def test_profile_enabled_function():
    """Test _jaxframe_profile_enabled returns correct values."""
    # Disabled by default
    if "JAXFRAME_PROFILE_TRANSFORM" in os.environ:
        del os.environ["JAXFRAME_PROFILE_TRANSFORM"]
    assert not _jaxframe_profile_enabled()
    
    # Enabled with "1"
    os.environ["JAXFRAME_PROFILE_TRANSFORM"] = "1"
    assert _jaxframe_profile_enabled()
    
    # Enabled with "true"
    os.environ["JAXFRAME_PROFILE_TRANSFORM"] = "true"
    assert _jaxframe_profile_enabled()
    
    # Enabled with "yes"
    os.environ["JAXFRAME_PROFILE_TRANSFORM"] = "yes"
    assert _jaxframe_profile_enabled()
    
    # Disabled with "0"
    os.environ["JAXFRAME_PROFILE_TRANSFORM"] = "0"
    assert not _jaxframe_profile_enabled()


def test_profile_limit_function():
    """Test _jaxframe_profile_limit returns correct values."""
    # Default value
    if "JAXFRAME_PROFILE_LIMIT" in os.environ:
        del os.environ["JAXFRAME_PROFILE_LIMIT"]
    assert _jaxframe_profile_limit() == 20
    
    # Custom value
    os.environ["JAXFRAME_PROFILE_LIMIT"] = "100"
    assert _jaxframe_profile_limit() == 100
    
    # Invalid value returns default
    os.environ["JAXFRAME_PROFILE_LIMIT"] = "invalid"
    assert _jaxframe_profile_limit() == 20


def test_should_profile_respects_limit():
    """Test that _should_profile respects the call limit."""
    os.environ["JAXFRAME_PROFILE_TRANSFORM"] = "1"
    os.environ["JAXFRAME_PROFILE_LIMIT"] = "3"
    
    # First 3 calls should be profiled
    assert _should_profile("test_func", False) is True  # Call 1
    assert _should_profile("test_func", False) is True  # Call 2
    assert _should_profile("test_func", False) is True  # Call 3
    
    # 4th call should return False (limit message)
    assert _should_profile("test_func", False) is False  # Call 4
    
    # Subsequent calls should also return False
    assert _should_profile("test_func", False) is False  # Call 5
    assert _should_profile("test_func", False) is False  # Call 6


def test_should_profile_separate_counters():
    """Test that different functions have separate call counters."""
    os.environ["JAXFRAME_PROFILE_TRANSFORM"] = "1"
    os.environ["JAXFRAME_PROFILE_LIMIT"] = "2"
    
    # Each function should have its own counter
    assert _should_profile("func_a", False) is True  # func_a: call 1
    assert _should_profile("func_b", False) is True  # func_b: call 1
    assert _should_profile("func_a", False) is True  # func_a: call 2
    assert _should_profile("func_b", False) is True  # func_b: call 2
    
    # Both should hit their limits independently
    assert _should_profile("func_a", False) is False  # func_a: call 3 (limit)
    assert _should_profile("func_b", False) is False  # func_b: call 3 (limit)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_profiling_doesnt_break_pivot():
    """Test that profiling doesn't break pivot operations."""
    # Enable profiling
    os.environ["JAXFRAME_PROFILE_TRANSFORM"] = "1"
    os.environ["JAXFRAME_PROFILE_LIMIT"] = "5"
    
    # Create test data
    long_df = DataFrame({
        'id': ['A', 'A', 'B', 'B'],
        'time': [0, 1, 0, 1],
        'value': jnp.array([1.0, 2.0, 3.0, 4.0])
    })
    
    # Pivot should work with profiling enabled
    wide_df = pivot_sparse(
        long_df,
        index='id',
        value='value',
        on='time',
        prefix='t'
    )
    
    # Check result is correct
    assert len(wide_df) == 2
    assert 't$0$value' in wide_df.columns
    assert 't$1$value' in wide_df.columns
    assert 't$0$mask' in wide_df.columns
    assert 't$1$mask' in wide_df.columns


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_profiling_doesnt_break_unpivot():
    """Test that profiling doesn't break unpivot operations."""
    # Enable profiling
    os.environ["JAXFRAME_PROFILE_TRANSFORM"] = "1"
    os.environ["JAXFRAME_PROFILE_LIMIT"] = "5"
    
    # Create wide test data
    wide_df = DataFrame({
        'id': ['A', 'B'],
        't$0$value': jnp.array([1.0, 3.0]),
        't$1$value': jnp.array([2.0, 4.0]),
        't$0$mask': [True, True],
        't$1$mask': [True, True]
    })
    
    # Unpivot should work with profiling enabled
    long_df = unpivot_sparse(
        wide_df,
        index='id',
        var_name='time',
        value_name='value'
    )
    
    # Check result is correct
    assert len(long_df) == 4
    assert 'id' in long_df.columns
    assert 'time' in long_df.columns
    assert 'value' in long_df.columns


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_profiling_doesnt_break_masked_array():
    """Test that profiling doesn't break masked array conversion."""
    # Enable profiling
    os.environ["JAXFRAME_PROFILE_TRANSFORM"] = "1"
    os.environ["JAXFRAME_PROFILE_LIMIT"] = "5"
    
    # Create wide test data
    wide_df = DataFrame({
        'id': ['A', 'B'],
        't$0$value': jnp.array([1.0, 3.0]),
        't$1$value': jnp.array([2.0, 4.0]),
        't$0$mask': [True, True],
        't$1$mask': [True, True]
    })
    
    # Convert to masked array - use correct pattern with 3 groups
    ma = to_masked_array(wide_df, index='id', pattern=r'([^$]+)\$(\d+)\$value')
    
    # Check result is correct
    assert ma.data.shape == (2, 2)
    assert ma.mask.shape == (2, 2)
    
    # Convert back to wide
    reconstructed = from_masked_array(ma, prefix='t')
    
    # Check reconstruction
    assert len(reconstructed) == 2
    assert 't$0$value' in reconstructed.columns
    assert 't$1$value' in reconstructed.columns


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_profiling_output_captured(capsys):
    """Test that profiling actually produces output when enabled."""
    # Enable profiling
    os.environ["JAXFRAME_PROFILE_TRANSFORM"] = "1"
    os.environ["JAXFRAME_PROFILE_LIMIT"] = "5"
    
    # Create test data with traced values inside a JIT context
    # to ensure the JAX path is taken
    import jax
    
    @jax.jit
    def do_pivot(values):
        """JIT-compiled function that performs pivot."""
        long_df = DataFrame({
            'id': ['A', 'A', 'B', 'B'],
            'time': [0, 1, 0, 1],
            'value': values
        })
        
        # Perform a pivot operation
        wide_df = pivot_sparse(
            long_df,
            index='id',
            value='value',
            on='time',
            prefix='t'
        )
        
        # Return a value to keep JAX happy
        return wide_df['t$0$value'][0]
    
    # Call the JIT-compiled function - this will trigger tracing
    result = do_pivot(jnp.array([1.0, 2.0, 3.0, 4.0]))
    
    # Capture output
    captured = capsys.readouterr()
    
    # Check that profiling output was produced
    # Note: During JIT compilation, the function is traced, so profiling should trigger
    assert '[JAXFRAME-TIMING]' in captured.out or '[JAXFRAME-TIMING]' in captured.err
    
    # If it's in err, print it for debugging
    if '[JAXFRAME-TIMING]' in captured.err:
        print("Profiling output found in stderr:", captured.err)


def test_profiling_disabled_by_default(capsys):
    """Test that profiling produces no output when disabled."""
    # Make sure profiling is disabled
    if "JAXFRAME_PROFILE_TRANSFORM" in os.environ:
        del os.environ["JAXFRAME_PROFILE_TRANSFORM"]
    
    # Create test data (without JAX arrays to avoid JAX dependency)
    long_df = DataFrame({
        'id': ['A', 'A', 'B', 'B'],
        'time': [0, 1, 0, 1],
        'value': [1.0, 2.0, 3.0, 4.0]
    })
    
    # Perform a pivot operation (will use non-JAX path)
    wide_df = pivot_sparse(
        long_df,
        index='id',
        value='value',
        on='time',
        prefix='t'
    )
    
    # Capture output
    captured = capsys.readouterr()
    
    # Check that NO profiling output was produced
    assert '[JAXFRAME-TIMING]' not in captured.out
    assert 'pivot_jax' not in captured.out


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
