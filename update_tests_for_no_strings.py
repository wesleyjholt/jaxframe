"""
Script to update tests to remove string shortcuts for aggregation functions.
"""
import re
import os

def create_helper_functions():
    """Return the helper functions to add at the top of test files."""
    return '''
# Helper aggregation functions (no string shortcuts)
import jax.numpy as jnp

def agg_sum(x):
    """Sum aggregation."""
    return jnp.sum(x)

def agg_mean(x):
    """Mean aggregation."""
    return jnp.mean(x)

def agg_std(x):
    """Standard deviation aggregation."""
    return jnp.std(x)

def agg_min(x):
    """Minimum aggregation."""
    return jnp.min(x)

def agg_max(x):
    """Maximum aggregation."""
    return jnp.max(x)

def agg_count(x):
    """Count aggregation."""
    return jnp.array(len(x), dtype=x.dtype)
'''

# Mapping from string shortcuts to named tuples
STRING_TO_TUPLE = {
    "'sum'": "('sum', agg_sum)",
    '"sum"': '("sum", agg_sum)',
    "'mean'": "('mean', agg_mean)",
    '"mean"': '("mean", agg_mean)',
    "'std'": "('std', agg_std)",
    '"std"': '("std", agg_std)',
    "'min'": "('min', agg_min)",
    '"min"': '("min", agg_min)',
    "'max'": "('max', agg_max)",
    '"max"': '("max", agg_max)',
    "'count'": "('count', agg_count)",
    '"count"': '("count", agg_count)',
}

def update_file(filepath):
    """Update a test file to use named tuples instead of string shortcuts."""
    print(f"Updating {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Add helper functions if not already present
    if 'def agg_sum(' not in content:
        # Find the imports section
        import_match = re.search(r'(from jaxframe import DataFrame\n)', content)
        if import_match:
            insert_pos = import_match.end()
            content = content[:insert_pos] + '\n' + create_helper_functions() + '\n' + content[insert_pos:]
    
    # Replace string shortcuts with named tuples
    for old, new in STRING_TO_TUPLE.items():
        content = content.replace(old, new)
    
    # Special case: update test that expects 'median' to not be supported
    # This should now expect TypeError about string not supported
    if 'test_unsupported_agg_function' in content:
        content = re.sub(
            r"with pytest\.raises\(ValueError, match=\"Unsupported aggregation function\"\):",
            "with pytest.raises(TypeError, match=\"String shortcuts.*are no longer supported\"):",
            content
        )
        content = re.sub(
            r"df\.group_by\('category'\)\.agg\(\{'value': \('median', agg_median\)\}\)",
            "df.group_by('category').agg({'value': 'median'})  # String should error",
            content
        )
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✓ Updated {filepath}")
        return True
    else:
        print(f"  - No changes needed for {filepath}")
        return False

def main():
    """Update all test files."""
    test_files = [
        'tests/test_group_by.py',
        'tests/test_groupby_apply.py',
        'tests/test_custom_agg.py',
        'tests/test_named_agg.py',
    ]
    
    updated_count = 0
    for filepath in test_files:
        if os.path.exists(filepath):
            if update_file(filepath):
                updated_count += 1
        else:
            print(f"Warning: {filepath} not found")
    
    print(f"\nUpdated {updated_count} files")

if __name__ == '__main__':
    main()
