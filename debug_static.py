#!/usr/bin/env python3

import jax
import jax.numpy as jnp
from jaxframe import DataFrame

# Create a simple DataFrame
df = DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
print(f"DataFrame hash: {hash(df)}")
print(f"DataFrame type: {type(df)}")

def test_func(arr, static_df):
    print(f"Inside function - df type: {type(static_df)}")
    return jnp.sum(arr)

# Try to JIT with static argument
print("Trying to create JIT function...")
try:
    jit_func = jax.jit(test_func, static_argnames=['static_df'])
    print("JIT function created successfully!")
    
    print("Calling JIT function...")
    result = jit_func(jnp.array([1.0, 2.0, 3.0]), df)
    print(f"Result: {result}")
    
except Exception as e:
    print(f"Error: {e}")
    print(f"Error type: {type(e)}")
