"""
Wide-to-Long Data Transformations

Shows JAXFrame's data reshaping capabilities:
- Wide format data with time series
- Mask-based missing data handling
- Wide-to-long transformations
- Long-to-wide conversions
- Multiple variable handling
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import jax.numpy as jnp
from src.jaxframe import DataFrame, wide_to_long_masked, long_to_wide_masked

def main():
    print("=== Wide-to-Long Data Transformations ===\n")
    
    # 1. Create wide format data (time series with missing values)
    print("1. Wide Format Time Series Data:")
    wide_data = {
        'patient_id': ['P001', 'P002', 'P003', 'P004'],
        'baseline_age': [25, 34, 45, 29],
        
        # Time series measurements (temperature)
        'temp$0$value': jnp.array([98.6, 99.1, 97.8, 98.2]),
        'temp$1$value': jnp.array([98.9, 99.5, 98.1, 98.7]),
        'temp$2$value': jnp.array([99.2, 99.8, 98.4, 98.9]),
        
        # Masks (True = valid measurement, False = missing)
        'temp$0$mask': np.array([True, True, True, True]),      # All baseline measurements
        'temp$1$mask': np.array([True, True, False, True]),     # P003 missing day 1
        'temp$2$mask': np.array([True, False, False, True]),    # P002, P003 missing day 2
    }
    
    wide_df = DataFrame(wide_data, name="patient_temps_wide")
    print(f"Wide format shape: {wide_df.shape}")
    print("Sample of wide data:")
    print(wide_df)
    print()
    
    # 2. Convert to long format (removes missing values)
    print("2. Wide-to-Long Conversion:")
    long_df = wide_to_long_masked(
        wide_df, 
        id_columns=['patient_id', 'baseline_age'],
        var_name='day',
        value_name='temperature'
    )
    
    print(f"Long format shape: {long_df.shape}")
    print("Long format data (missing values filtered out):")
    print(long_df)
    print()
    
    # 3. Multiple variables example
    print("3. Multiple Variables Wide-to-Long:")
    multi_wide_data = {
        'subject_id': ['S1', 'S2', 'S3'],
        
        # Heart rate measurements
        'hr$0$value': jnp.array([72.0, 68.0, 75.0]),
        'hr$1$value': jnp.array([74.0, 70.0, 77.0]),
        'hr$0$mask': np.array([True, True, True]),
        'hr$1$mask': np.array([True, False, True]),  # S2 missing
        
        # Blood pressure measurements  
        'bp$0$value': jnp.array([120.0, 115.0, 125.0]),
        'bp$1$value': jnp.array([122.0, 118.0, 127.0]),
        'bp$0$mask': np.array([True, True, True]),
        'bp$1$mask': np.array([False, True, True]),  # S1 missing
    }
    
    multi_wide_df = DataFrame(multi_wide_data)
    
    # Convert each variable separately, then we could join them
    hr_long = wide_to_long_masked(
        multi_wide_df, 
        'subject_id',
        var_pattern=r'(hr)\$(\d+)\$(value|mask)',
        var_name='timepoint',
        value_name='heart_rate'
    )
    
    bp_long = wide_to_long_masked(
        multi_wide_df,
        'subject_id', 
        var_pattern=r'(bp)\$(\d+)\$(value|mask)',
        var_name='timepoint',
        value_name='blood_pressure'
    )
    
    print("Heart rate long format:")
    print(hr_long)
    print()
    print("Blood pressure long format:")
    print(bp_long)
    print()
    
    # 4. Long-to-wide conversion
    print("4. Long-to-Wide Conversion:")
    
    # Create some long format data
    long_sample = DataFrame({
        'id': ['A', 'A', 'B', 'B', 'C'],
        'time': [0, 1, 0, 2, 1],  # Note: B missing time 1, C missing times 0&2
        'measurement': jnp.array([1.2, 1.8, 2.1, 2.9, 3.4])
    })
    
    print("Original long format:")
    print(long_sample)
    print()
    
    # Convert back to wide
    wide_reconstructed = long_to_wide_masked(
        long_sample,
        id_columns='id',
        var_column='time',
        value_column='measurement',
        var_prefix='measure'
    )
    
    print("Reconstructed wide format:")
    print(wide_reconstructed)
    print()
    
    # 5. Demonstrate mask behavior
    print("5. Mask Behavior Analysis:")
    print("Original wide format masks:")
    for col in ['temp$0$mask', 'temp$1$mask', 'temp$2$mask']:
        mask_values = wide_df[col]
        valid_count = np.sum(mask_values)
        print(f"{col}: {mask_values} ({valid_count}/{len(mask_values)} valid)")
    
    print(f"\\nLong format total observations: {len(long_df)}")
    print("vs. if no missing data:", len(wide_df) * 3, "observations")
    
    # Show which observations were kept
    long_dict = long_df.to_dict()
    print("\\nObservations kept in long format:")
    for i in range(len(long_df)):
        pid = long_dict['patient_id'][i]
        day = long_dict['day'][i] 
        temp = long_dict['temperature'][i]
        print(f"  {pid} day {day}: {temp:.1f}°F")

if __name__ == "__main__":
    main()