"""
Convert bare callables to tuples in test files.
"""
import re

def convert_file(filepath):
    """Convert bare callables to named tuples in aggregation calls."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    # Pattern 1: Single bare callable -> wrap in tuple
    # .agg({'value': jnp.median}) -> .agg({'value': ('median', jnp.median)})
    content = re.sub(
        r"\.agg\(\{([^}]+):\s*jnp\.(\w+)\}\)",
        lambda m: f".agg({{{m.group(1)}: ('{m.group(2)}', jnp.{m.group(2)})}}})",
        content
    )
    
    # Pattern 2: Single bare function reference -> wrap in tuple  
    # .agg({'value': range_func}) -> .agg({'value': ('range_func', range_func)})
    # But exclude things already in tuples
    content = re.sub(
        r"\.agg\(\{([^}]+):\s*([a-z_]\w+)\}\)",
        lambda m: f".agg({{{m.group(1)}: ('{m.group(2)}', {m.group(2)})}}})",
        content
    )
    
    # Pattern 3: List with bare callables
    # [jnp.median, jnp.std] -> [('median', jnp.median), ('std', jnp.std)]
    def replace_list_callables(match):
        items = match.group(1).split(',')
        new_items = []
        for item in items:
            item = item.strip()
            # Skip if already a tuple
            if item.startswith('('):
                new_items.append(item)
            # Handle jnp.function
            elif item.startswith('jnp.'):
                func_name = item.split('.')[1]
                new_items.append(f"('{func_name}', {item})")
            # Handle bare function names
            elif re.match(r'^[a-z_]\w+$', item):
                new_items.append(f"('{item}', {item})")
            else:
                new_items.append(item)
        return '[' + ', '.join(new_items) + ']'
    
    content = re.sub(r'\[([^\]]+jnp\.[^\]]+)\]', replace_list_callables, content)
    
    # Pattern 4: Lambda functions -> wrap with descriptive name
    # lambda x: ... -> ('custom', lambda x: ...)
    content = re.sub(
        r"\.agg\(\{([^}]+):\s*(lambda\s+[^}]+)\}\)",
        lambda m: f".agg({{{m.group(1)}: ('custom', {m.group(2)})}}})",
        content
    )
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✓ Updated {filepath}")
        return True
    else:
        print(f"- No changes for {filepath}")
        return False

# Update test files
files = [
    'tests/test_custom_agg.py',
]

for f in files:
    convert_file(f)
