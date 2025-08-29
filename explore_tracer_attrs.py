import jax
import jax.numpy as jnp

# Let's explore tracer attributes
@jax.jit  
def explore_tracer():
    x = jnp.array([1.0, 2.0, 3.0])
    print("Type:", type(x))
    print("Has shape:", hasattr(x, 'shape'))
    print("Has dtype:", hasattr(x, 'dtype'))
    print("Shape:", x.shape)
    print("Dtype:", x.dtype)
    
    # Try to see what attributes are available
    print("Dir:", [attr for attr in dir(x) if not attr.startswith('_')])
    
    return x

print("Exploring tracer attributes...")
result = explore_tracer()
print("Final result:", result)
