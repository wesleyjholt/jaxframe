"""
Demonstration of the new MaskedArray functionality in jaxframe.

This example shows how to:
1. Create a MaskedArray from a wide DataFrame
2. Use the MaskedArray's methods and properties
3. Convert back to a wide DataFrame
4. Use the new API with masked data
"""

import jax.numpy as jnp
import numpy as np
from jaxframe import DataFrame, MaskedArray, wide_df_to_masked_array, masked_array_to_wide_df

def main():
    print("=== JAXFrame MaskedArray Demo ===\n")
    
    # Create a sample wide DataFrame with JAX arrays and masks
    print("1. Creating a wide DataFrame with JAX data and masks:")
    wide_data = {
        'patient_id': ['P001', 'P002', 'P003'],
        'measurement$0$value': jnp.array([1.2, 2.3, 3.4]),
        'measurement$1$value': jnp.array([1.5, 2.8, 4.1]), 
        'measurement$2$value': jnp.array([1.8, 3.1, 4.5]),
        'measurement$0$mask': np.array([True, True, False]),  # P003 missing first measurement
        'measurement$1$mask': np.array([True, False, True]),  # P002 missing second measurement
        'measurement$2$mask': np.array([False, True, True])   # P001 missing third measurement
    }
    wide_df = DataFrame(wide_data)
    print(f"Wide DataFrame shape: {len(wide_df)} rows, {len(wide_df.columns)} columns")
    print(f"Columns: {list(wide_df.columns)}")
    print()
    
    # Convert to MaskedArray using the new API
    print("2. Converting to MaskedArray:")
    masked_array = wide_df_to_masked_array(wide_df, 'patient_id')
    print(f"MaskedArray shape: {masked_array.shape}")
    print(f"Data type: {type(masked_array.data)} with shape {masked_array.data.shape}")
    print(f"Mask type: {type(masked_array.mask)} with shape {masked_array.mask.shape}")
    print(f"Index DataFrame columns: {list(masked_array.index_df.columns)}")
    print()
    
    # Demonstrate MaskedArray methods
    print("3. Using MaskedArray methods:")
    print(f"Number of valid data points: {len(masked_array.get_valid_data())}")
    valid_data = masked_array.get_valid_data()
    print(f"Valid data values: {valid_data}")
    print(f"Mean of valid data: {jnp.mean(valid_data):.2f}")
    print()
    
    # Show the data and mask arrays
    print("4. Data and mask details:")
    print("Data array:")
    print(masked_array.data)
    print("Mask array (True = valid, False = missing):")
    print(masked_array.mask)
    print("Index DataFrame:")
    print(masked_array.index_df.to_dict())
    print()
    
    # Create a copy and modify it
    print("5. Creating a copy and demonstrating immutability:")
    copy_array = masked_array.copy()
    print(f"Original and copy are equal: {masked_array == copy_array}")
    print(f"Original and copy are same object: {masked_array is copy_array}")
    print()
    
    # Convert back to wide DataFrame
    print("6. Converting back to wide DataFrame:")
    reconstructed_df = masked_array_to_wide_df(masked_array, var_prefix='recon')
    print(f"Reconstructed DataFrame shape: {len(reconstructed_df)} rows, {len(reconstructed_df.columns)} columns")
    print(f"Columns: {list(reconstructed_df.columns)}")
    
    # Verify that values are preserved
    print("\\n7. Verifying data preservation:")
    original_values = wide_df['measurement$0$value']
    reconstructed_values = reconstructed_df['recon$0$value']
    print(f"Values preserved: {jnp.allclose(original_values, reconstructed_values)}")
    
    original_masks = wide_df['measurement$0$mask']
    reconstructed_masks = reconstructed_df['recon$0$mask']
    print(f"Masks preserved: {np.array_equal(original_masks, reconstructed_masks)}")
    print()
    
    # Demonstrate string representation
    print("8. String representation:")
    print(masked_array)
    print()
    
    # Show serialization capability
    print("9. Serialization to dictionary:")
    array_dict = masked_array.to_dict()
    print(f"Dictionary keys: {list(array_dict.keys())}")
    print(f"Shape in dict: {array_dict['shape']}")
    print()
    
    print("=== Demo Complete ===")
    print("\\nKey benefits of MaskedArray:")
    print("• Encapsulates JAX data, numpy masks, and index metadata in one object")
    print("• Maintains proper data types (JAX for values, numpy for masks)")
    print("• Provides convenient methods for working with masked data")
    print("• Simplifies function signatures by returning single object instead of tuple")
    print("• Ensures data/mask/index consistency through validation")

if __name__ == "__main__":
    main()
