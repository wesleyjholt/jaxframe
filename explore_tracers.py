import jax
import jax.numpy as jnp
import equinox as eqx

# Create a simple example to see how equinox handles tracers
@jax.jit
def test_tracer_printing():
    x = jnp.array([1.0, 2.0, 3.0])
    y = x * 2
    print("Regular print:", x)
    print("String conversion:", str(x))
    print("Repr:", repr(x))
    
    # See what equinox tree_pprint does with tracers
    eqx.tree_pprint(x)
    eqx.tree_pprint({"data": x, "result": y})
    
    return y

print("Running JIT function to see tracer behavior...")
result = test_tracer_printing()
print("Result:", result)
