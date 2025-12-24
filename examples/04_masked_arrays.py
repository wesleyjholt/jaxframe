"""
MaskedArray Operations - Scientific Data Workflows

Shows MaskedArray functionality for scientific computing:
- Creating MaskedArrays from wide DataFrames
- JAX computations on masked data
- Statistical operations with missing values
- Roundtrip conversions preserving computational graphs
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import jax
import jax.numpy as jnp
from src.jaxframe import DataFrame, MaskedArray, wide_df_to_masked_array, masked_array_to_wide_df

def main():
    print("=== MaskedArray Scientific Workflows ===\n")
    
    # 1. Create experimental data with missing observations
    print("1. Experimental Data Setup:")
    np.random.seed(42)
    jax_key = jax.random.PRNGKey(42)
    
    # Simulate 5 experiments, 4 time points each
    n_experiments = 5
    n_timepoints = 4
    
    # Generate base measurements with noise
    base_values = jnp.linspace(1.0, 3.0, n_experiments)[:, None]
    time_trend = jnp.linspace(0.0, 1.0, n_timepoints)[None, :]
    true_data = base_values + 0.5 * time_trend
    noise = 0.1 * jax.random.normal(jax_key, (n_experiments, n_timepoints))
    measurements = true_data + noise
    
    # Create realistic missing data pattern
    missing_prob = np.array([0.0, 0.1, 0.2, 0.3])  # More missing at later times
    masks = np.random.rand(n_experiments, n_timepoints) > missing_prob[None, :]
    
    # Create wide format DataFrame
    wide_data = {
        'experiment_id': [f'E{i+1:03d}' for i in range(n_experiments)],
        'group': ['A', 'B', 'A', 'B', 'A'],  # Experimental groups
    }
    
    # Add value and mask columns
    for t in range(n_timepoints):
        wide_data[f'measurement${t}$value'] = measurements[:, t]
        wide_data[f'measurement${t}$mask'] = masks[:, t]
    
    wide_df = DataFrame(wide_data, name="experiment_data")
    print(f"Wide DataFrame: {wide_df.shape}")
    print("Sample data:")
    print(wide_df)
    print()
    
    # 2. Convert to MaskedArray for scientific computing
    print("2. MaskedArray Creation:")
    masked_array = wide_df_to_masked_array(
        wide_df, 
        id_columns=['experiment_id', 'group']
    )
    
    print(f"MaskedArray shape: {masked_array.shape}")
    print(f"Data type: {type(masked_array.data)}")
    print(f"Mask type: {type(masked_array.mask)}")
    print(f"Valid observations: {np.sum(masked_array.mask)}/{masked_array.mask.size}")
    print()
    
    print("Data matrix:")
    print(masked_array.data)
    print("Mask matrix (True = valid):")
    print(masked_array.mask)
    print()
    
    # 3. Statistical computations with missing data
    print("3. Statistical Analysis:")
    
    # Get only valid data points
    valid_data = masked_array.get_valid_data()
    print(f"Valid data points: {len(valid_data)}")
    print(f"Mean of valid data: {jnp.mean(valid_data):.4f}")
    print(f"Std of valid data: {jnp.std(valid_data):.4f}")
    print()
    
    # Per-experiment statistics (handling missing values)
    print("Per-experiment analysis:")
    for i in range(n_experiments):
        exp_data = masked_array.data[i]
        exp_mask = masked_array.mask[i]
        exp_valid = exp_data[exp_mask]  # Only valid measurements
        
        if len(exp_valid) > 0:
            exp_mean = jnp.mean(exp_valid)
            exp_id = masked_array.index_df['experiment_id'][i]
            exp_group = masked_array.index_df['group'][i]
            n_valid = len(exp_valid)
            print(f"  {exp_id} (Group {exp_group}): {exp_mean:.3f} (n={n_valid})")
    print()
    
    # 4. JAX computations preserving computational graph
    print("4. Differentiable Computations:")
    
    def model_loss(data_matrix, target_trend):
        """Compute loss comparing data to expected trend, handling masks."""
        # Simple model: predict linear trend for each experiment
        n_exp, n_time = data_matrix.shape
        time_points = jnp.linspace(0, 1, n_time)
        
        # Predict values for each experiment
        predicted = target_trend[:, None] * time_points[None, :]
        
        # Compute MSE loss (this ignores masking for simplicity)
        return jnp.mean((data_matrix - predicted) ** 2)
    
    # Optimize trend parameters
    def loss_fn(trend_params):
        return model_loss(masked_array.data, trend_params)
    
    # Initial guess for trend parameters (one per experiment)
    initial_trends = jnp.ones(n_experiments)
    
    # Compute gradient
    grad_fn = jax.grad(loss_fn)
    loss_val = loss_fn(initial_trends)
    gradient = grad_fn(initial_trends)
    
    print(f"Initial loss: {loss_val:.6f}")
    print(f"Gradient shape: {gradient.shape}")
    print(f"Gradient values: {gradient}")
    print()
    
    # 5. Roundtrip conversion preserving computational graph
    print("5. Roundtrip Conversion Test:")
    
    def roundtrip_loss(data_matrix):
        """Test that roundtrip conversion preserves gradients."""
        # Convert to wide DataFrame
        temp_wide = masked_array_to_wide_df(
            MaskedArray(data_matrix, masked_array.mask, masked_array.index_df),
            var_prefix='temp'
        )
        
        # Convert back to MaskedArray
        temp_masked = wide_df_to_masked_array(temp_wide, ['experiment_id', 'group'])
        
        # Simple computation on the roundtrip data
        return jnp.sum(temp_masked.data ** 2)
    
    # Test gradient computation through roundtrip
    roundtrip_grad_fn = jax.grad(roundtrip_loss)
    test_data = masked_array.data
    
    original_loss = jnp.sum(test_data ** 2)
    roundtrip_loss_val = roundtrip_loss(test_data)
    roundtrip_grad = roundtrip_grad_fn(test_data)
    
    print(f"Original computation: {original_loss:.6f}")
    print(f"Roundtrip computation: {roundtrip_loss_val:.6f}")
    print(f"Values match: {jnp.allclose(original_loss, roundtrip_loss_val)}")
    print(f"Gradient computation successful: {roundtrip_grad.shape == test_data.shape}")
    print()
    
    # 6. Group-wise analysis
    print("6. Group-wise Analysis:")
    groups = np.array(masked_array.index_df['group'])
    
    for group in ['A', 'B']:
        group_mask = groups == group
        group_indices = np.where(group_mask)[0]
        
        # Extract data for this group
        group_data = masked_array.data[group_indices]
        group_masks = masked_array.mask[group_indices]
        
        # Get all valid measurements for this group
        group_valid = group_data[group_masks]
        
        if len(group_valid) > 0:
            group_mean = jnp.mean(group_valid)
            group_n = len(group_valid)
            print(f"Group {group}: mean = {group_mean:.4f} (n={group_n} observations)")
    
    print()
    print("7. MaskedArray Properties:")
    print(f"Shape: {masked_array.shape}")
    print(f"Index DataFrame columns: {masked_array.index_df.columns}")
    print(f"Data dtype: {masked_array.data.dtype}")
    print(f"Mask dtype: {masked_array.mask.dtype}")
    
    # Test equality
    copy_array = masked_array.copy()
    print(f"Copy equals original: {masked_array == copy_array}")
    print(f"Copy is same object: {masked_array is copy_array}")

if __name__ == "__main__":
    main()