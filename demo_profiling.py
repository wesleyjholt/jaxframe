"""
Demonstration of jaxframe profiling capabilities.

This script shows how to enable and use profiling to understand
JAX compilation behavior in jaxframe transform operations.

Usage:
    # Enable profiling for first 10 calls
    JAXFRAME_PROFILE_TRANSFORM=1 JAXFRAME_PROFILE_LIMIT=10 python demo_profiling.py
    
    # Disable profiling (default)
    python demo_profiling.py
"""

import os
import sys

# Add src to path if running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from jaxframe import DataFrame
from jaxframe.transform import pivot_sparse, unpivot_sparse, to_masked_array, from_masked_array

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    print("JAX not available - please install with: pip install jax")
    sys.exit(1)


def demo_pivot_profiling():
    """Demonstrate profiling of pivot operations."""
    print("\n" + "="*70)
    print("DEMO 1: Pivot Operation Profiling")
    print("="*70)
    
    # Create sample long-format data
    long_df = DataFrame({
        'patient_id': ['P1', 'P1', 'P1', 'P2', 'P2', 'P2'],
        'time': [0, 1, 2, 0, 1, 2],
        'concentration': jnp.array([0.0, 10.5, 8.2, 0.0, 12.3, 9.1])
    })
    
    print("\nInput long-format data:")
    print(f"  Rows: {len(long_df)}")
    print(f"  Columns: {long_df.columns}")
    
    print("\nPerforming pivot operation...")
    
    # Perform pivot - this will show profiling if enabled
    wide_df = pivot_sparse(
        long_df,
        index='patient_id',
        value='concentration',
        on='time',
        prefix='conc'
    )
    
    print(f"\nOutput wide-format data:")
    print(f"  Rows: {len(wide_df)}")
    print(f"  Columns: {wide_df.columns}")
    print("\nNote: If JAXFRAME_PROFILE_TRANSFORM=1, you should see timing output above.")


def demo_unpivot_profiling():
    """Demonstrate profiling of unpivot operations."""
    print("\n" + "="*70)
    print("DEMO 2: Unpivot Operation Profiling")
    print("="*70)
    
    # Create sample wide-format data
    wide_df = DataFrame({
        'patient_id': ['P1', 'P2'],
        'conc$0$value': jnp.array([0.0, 0.0]),
        'conc$1$value': jnp.array([10.5, 12.3]),
        'conc$2$value': jnp.array([8.2, 9.1]),
        'conc$0$mask': [True, True],
        'conc$1$mask': [True, True],
        'conc$2$mask': [True, True],
    })
    
    print("\nInput wide-format data:")
    print(f"  Rows: {len(wide_df)}")
    print(f"  Columns: {wide_df.columns}")
    
    print("\nPerforming unpivot operation...")
    
    # Perform unpivot - this will show profiling if enabled
    long_df = unpivot_sparse(
        wide_df,
        index='patient_id',
        var_name='time',
        value_name='concentration'
    )
    
    print(f"\nOutput long-format data:")
    print(f"  Rows: {len(long_df)}")
    print(f"  Columns: {long_df.columns}")
    print("\nNote: If JAXFRAME_PROFILE_TRANSFORM=1, you should see timing output above.")


def demo_masked_array_profiling():
    """Demonstrate profiling of masked array conversions."""
    print("\n" + "="*70)
    print("DEMO 3: Masked Array Conversion Profiling")
    print("="*70)
    
    # Create sample wide-format data
    wide_df = DataFrame({
        'patient_id': ['P1', 'P2'],
        'conc$0$value': jnp.array([0.0, 0.0]),
        'conc$1$value': jnp.array([10.5, 12.3]),
        'conc$2$value': jnp.array([8.2, 9.1]),
        'conc$0$mask': [True, True],
        'conc$1$mask': [True, True],
        'conc$2$mask': [True, True],
    })
    
    print("\nInput wide-format DataFrame:")
    print(f"  Rows: {len(wide_df)}")
    print(f"  Columns: {wide_df.columns}")
    
    print("\nConverting to MaskedArray...")
    
    # Convert to masked array - this will show profiling if enabled
    ma = to_masked_array(
        wide_df,
        index='patient_id',
        pattern=r'([^$]+)\$(\d+)\$value'
    )
    
    print(f"\nOutput MaskedArray:")
    print(f"  Data shape: {ma.data.shape}")
    print(f"  Mask shape: {ma.mask.shape}")
    
    print("\nConverting back to DataFrame...")
    
    # Convert back to DataFrame
    reconstructed = from_masked_array(ma, prefix='conc')
    
    print(f"\nReconstructed DataFrame:")
    print(f"  Rows: {len(reconstructed)}")
    print(f"  Columns: {reconstructed.columns}")
    print("\nNote: If JAXFRAME_PROFILE_TRANSFORM=1, you should see timing output above.")


def demo_jit_compilation():
    """Demonstrate profiling during JIT compilation."""
    print("\n" + "="*70)
    print("DEMO 4: Profiling During JIT Compilation")
    print("="*70)
    
    @jax.jit
    def process_data(concentrations):
        """JIT-compiled function that performs pivot."""
        long_df = DataFrame({
            'patient_id': ['P1', 'P1', 'P1', 'P2', 'P2', 'P2'],
            'time': [0, 1, 2, 0, 1, 2],
            'concentration': concentrations
        })
        
        # Pivot operation inside JIT
        wide_df = pivot_sparse(
            long_df,
            index='patient_id',
            value='concentration',
            on='time',
            prefix='conc'
        )
        
        # Return something concrete
        return wide_df['conc$0$value'][0]
    
    print("\nCalling JIT-compiled function (first call - will trace)...")
    result1 = process_data(jnp.array([0.0, 10.5, 8.2, 0.0, 12.3, 9.1]))
    print(f"Result: {result1}")
    
    print("\nCalling JIT-compiled function (second call - uses cached compilation)...")
    result2 = process_data(jnp.array([0.0, 11.0, 8.5, 0.0, 13.0, 9.5]))
    print(f"Result: {result2}")
    
    print("\nNote: If JAXFRAME_PROFILE_TRANSFORM=1, you should see:")
    print("  - Profiling output on first call (during tracing)")
    print("  - traced=True in the profiling output")
    print("  - No profiling output on second call (unless within JAXFRAME_PROFILE_LIMIT)")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("JAXFRAME PROFILING DEMONSTRATION")
    print("="*70)
    
    # Check if profiling is enabled
    profiling_enabled = os.environ.get("JAXFRAME_PROFILE_TRANSFORM", "0") in {"1", "true", "True", "yes", "YES"}
    profile_limit = os.environ.get("JAXFRAME_PROFILE_LIMIT", "20")
    
    print(f"\nProfiling status:")
    print(f"  JAXFRAME_PROFILE_TRANSFORM: {os.environ.get('JAXFRAME_PROFILE_TRANSFORM', '(not set)')}")
    print(f"  JAXFRAME_PROFILE_LIMIT: {profile_limit}")
    print(f"  Profiling enabled: {profiling_enabled}")
    
    if not profiling_enabled:
        print("\n" + "*"*70)
        print("PROFILING IS DISABLED")
        print("To enable profiling, run:")
        print("  JAXFRAME_PROFILE_TRANSFORM=1 python demo_profiling.py")
        print("*"*70)
    
    # Run demonstrations
    demo_pivot_profiling()
    demo_unpivot_profiling()
    demo_masked_array_profiling()
    demo_jit_compilation()
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    
    if profiling_enabled:
        print("\nYou should have seen timing output above for each operation.")
        print("The profiling shows:")
        print("  - Function name and call number")
        print("  - Whether values are being traced (JAX compilation)")
        print("  - Timing for subphases (structure, extraction, application)")
        print("  - Total time for each operation")
    else:
        print("\nTo see profiling output, enable it with:")
        print("  JAXFRAME_PROFILE_TRANSFORM=1 python demo_profiling.py")
        print("\nTo limit output to first N calls:")
        print("  JAXFRAME_PROFILE_TRANSFORM=1 JAXFRAME_PROFILE_LIMIT=5 python demo_profiling.py")
    
    print()


if __name__ == '__main__':
    main()
