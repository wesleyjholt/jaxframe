#!/usr/bin/env python3

import jax
import jax.numpy as jnp
from jaxframe import DataFrame
from jaxframe.jitprint import jit_print_dataframe, _format_value_for_jit_print

def create_artificial_complex_tracer():
    """Try to create the exact type of complex tracer that was problematic."""
    
    print("=== Creating Artificial Complex Tracer ===")
    
    # Create a mock tracer with the same characteristics as the problematic one
    class MockComplexTracer:
        def __init__(self):
            self.aval = None  # Make aval extraction fail initially
        
        def __str__(self):
            # Return the exact problematic string you encountered
            return ("Traced<ShapedArray(float32[])>with<JVPTrace> with\n  primal = Array(0.11994141, dtype=float32)\n  "
                    "tangent = Traced<ShapedArray(float32[])>with<JaxprTrace> with\n    pval = (ShapedArray(float32[]), None)\n    "
                    "recipe = JaxprEqnRecipe(eqn_id=<object object at 0x1681ea540>, in_tracers=(Traced<ShapedArray(float32[1]):JaxprTrace>,), "
                    "out_tracer_refs=[<weakref at 0x1681f4f40; to 'JaxprTracer' at 0x1681f4550>], out_avals=[ShapedArray(float32[])], "
                    "primitive=squeeze, params={'dimensions': (0,)}, effects=frozenset(), "
                    "source_info=<jax._src.source_info_util.SourceInfo object at 0x30f49a170>, "
                    "ctx=JaxprEqnContext(compute_type=None, threefry_partitionable=True, cur_abstract_mesh=AbstractMesh((), axis_types=()), xla_metadata=None))")
    
    # Test our formatter on this mock tracer
    mock_tracer = MockComplexTracer()
    
    print(f"Mock tracer string length: {len(str(mock_tracer))}")
    print(f"First 200 chars: {str(mock_tracer)[:200]}...")
    
    formatted = _format_value_for_jit_print(mock_tracer)
    print(f"Our formatter result: '{formatted}'")
    
    # Now test with real JAX operations that might create similar tracers
    print("\n=== Testing Real Complex JAX Operations ===")
    
    def very_complex_function(x):
        # Multiple operations that could create complex nested tracers
        y = x.reshape(-1, 1)
        z = jnp.squeeze(y, axis=1)
        w = jnp.expand_dims(z, axis=0)
        final = jnp.squeeze(w, axis=0)
        return final
    
    # Apply multiple transformations
    grad_fn = jax.grad(lambda x: jnp.sum(very_complex_function(x) ** 2))
    jvp_fn = lambda x, v: jax.jvp(grad_fn, (x,), (v,))
    
    @jax.jit
    def test_complex_ops():
        x = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([0.1, 0.2, 0.3])
        
        # This should create very complex tracers
        primal, tangent = jvp_fn(x, v)
        
        jax.debug.print("Complex primal format: {}", _format_value_for_jit_print(primal))
        jax.debug.print("Complex tangent format: {}", _format_value_for_jit_print(tangent))
        
        return primal, tangent
    
    try:
        primal, tangent = test_complex_ops()
        print(f"Real complex operations completed successfully")
        print(f"Primal shape: {primal.shape}, Tangent shape: {tangent.shape}")
        return True
    except Exception as e:
        print(f"Error in complex operations: {e}")
        return False

def test_emergency_catch_all():
    """Test the emergency catch-all detection."""
    
    print("\n=== Testing Emergency Catch-All ===")
    
    # Create a value that should trigger the catch-all
    class ProblematicTracer:
        def __init__(self):
            # Add some aval for extraction
            class MockAval:
                def __init__(self):
                    self.dtype = 'float32'
                    self.shape = ()
            self.aval = MockAval()
        
        def __str__(self):
            # Long string with tracer indicators
            return ("Traced<ShapedArray(float32[])>with<JVPTrace> with lots of extra information " * 10)
    
    problematic = ProblematicTracer()
    print(f"Problematic tracer string length: {len(str(problematic))}")
    
    formatted = _format_value_for_jit_print(problematic)
    print(f"Emergency catch-all result: '{formatted}'")
    
    # Test with no aval
    class ProblematicTracerNoAval:
        def __str__(self):
            return ("Traced<ShapedArray(float32[])>with<JVPTrace> with lots of extra information " * 10)
    
    problematic_no_aval = ProblematicTracerNoAval()
    formatted_no_aval = _format_value_for_jit_print(problematic_no_aval)
    print(f"Emergency catch-all (no aval) result: '{formatted_no_aval}'")

if __name__ == "__main__":
    create_artificial_complex_tracer()
    test_emergency_catch_all()
    
    print("\n✅ All catch-all tests completed!")
