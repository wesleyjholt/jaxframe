"""
JAX Integration - JIT Compilation and Gradients

Shows how JAXFrame maintains JAX computational graphs:
- JAX arrays in DataFrames
- JIT compilation with DataFrames  
- Gradient computation through DataFrame operations
- Tracer handling during compilation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
from src.jaxframe import DataFrame

def main():
    print("=== JAX Integration Examples ===\n")
    
    # 1. JAX arrays in DataFrames
    print("1. JAX Arrays in DataFrames:")
    params = jnp.array([2.0, 3.0, 1.5])
    targets = jnp.array([4.1, 5.8, 2.9])
    
    data = {
        'id': ['exp1', 'exp2', 'exp3'],
        'params': params,
        'targets': targets,
        'weights': jnp.array([1.0, 2.0, 0.5])
    }
    
    df = DataFrame(data, name="experiments")
    print(df)
    print()
    
    # 2. Simple computation preserving gradients
    print("2. Gradient Computation:")
    
    def loss_fn(param_values):
        # Create DataFrame inside function - computational graph preserved
        df_internal = DataFrame({
            'id': ['a', 'b', 'c'],
            'params': param_values,
            'targets': jnp.array([4.0, 6.0, 3.0])
        })
        
        # DataFrame operations maintain differentiability
        params = df_internal['params']
        targets = df_internal['targets']
        
        # Simple MSE loss
        return jnp.mean((params - targets) ** 2)
    
    # Compute gradient
    grad_fn = jax.grad(loss_fn)
    test_params = jnp.array([2.0, 3.0, 1.0])
    
    loss_val = loss_fn(test_params)
    gradient = grad_fn(test_params)
    
    print(f"Loss: {loss_val:.4f}")
    print(f"Gradient: {gradient}")
    print()
    
    # 3. JIT compilation
    print("3. JIT Compilation:")
    
    @jax.jit
    def compute_weighted_sum(param_array, weight_array):
        # DataFrames with JAX arrays work inside JIT
        df_jit = DataFrame({
            'params': param_array,
            'weights': weight_array
        })
        
        return jnp.sum(df_jit['params'] * df_jit['weights'])
    
    # First call compiles
    result1 = compute_weighted_sum(jnp.array([1.0, 2.0]), jnp.array([0.5, 1.5]))
    print(f"JIT result 1: {result1}")
    
    # Second call uses compiled version
    result2 = compute_weighted_sum(jnp.array([3.0, 4.0]), jnp.array([0.2, 0.8]))
    print(f"JIT result 2: {result2}")
    print()
    
    # 4. Tracer demonstration
    print("4. Tracer Handling:")
    
    def create_df_with_tracers(x):
        # Inside JIT/grad, x becomes a tracer
        df_tracer = DataFrame({
            'tracer_col': x,
            'constant': jnp.array([1.0, 2.0])
        })
        
        # DataFrame can handle tracers without crashing
        print(f"DataFrame created with tracer successfully")
        return jnp.sum(df_tracer['tracer_col'] * df_tracer['constant'])
    
    # This should work without ConcretizationTypeError
    try:
        jit_fn = jax.jit(create_df_with_tracers)
        result = jit_fn(jnp.array([3.0, 4.0]))
        print(f"Tracer handling successful: {result}")
    except Exception as e:
        print(f"Error with tracers: {e}")
    print()
    
    # 5. Real optimization example
    print("5. Optimization Example:")
    
    # Generate synthetic data
    true_params = jnp.array([2.5, -1.2])
    X = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]])
    y = X @ true_params + 0.1 * jax.random.normal(jax.random.PRNGKey(0), (3,))
    
    def model_loss(params):
        # Use DataFrame for organized data handling
        model_df = DataFrame({
            'predictions': X @ params,
            'targets': y
        })
        
        pred = model_df['predictions']
        target = model_df['targets']
        return jnp.mean((pred - target) ** 2)
    
    # Optimize using JAX
    grad_fn = jax.grad(model_loss)
    
    # Simple gradient descent
    params = jnp.array([0.0, 0.0])
    learning_rate = 0.1
    
    for i in range(50):
        loss = model_loss(params)
        gradient = grad_fn(params)
        params = params - learning_rate * gradient
        
        if i % 10 == 0:
            print(f"Step {i}: Loss = {loss:.6f}, Params = {params}")
    
    print(f"True params: {true_params}")
    print(f"Final params: {params}")
    print(f"Parameter error: {jnp.linalg.norm(params - true_params):.6f}")

if __name__ == "__main__":
    main()