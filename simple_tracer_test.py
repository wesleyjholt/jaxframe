import jax
import jax.numpy as jnp
from jaxframe.jitprint import _format_value_for_jit_print

# Test the helper function
@jax.jit
def simple_test():
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1, 2, 3])
    z = jnp.array(5.0)
    
    jax.debug.print("Testing tracer formatting:")
    jax.debug.print("Float array: {}", _format_value_for_jit_print(x))
    jax.debug.print("Int array: {}", _format_value_for_jit_print(y))
    jax.debug.print("Scalar: {}", _format_value_for_jit_print(z))
    
    return x

print("Running simple test...")
result = simple_test()
print("Test completed, result:", result)
