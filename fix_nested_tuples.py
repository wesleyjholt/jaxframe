"""Fix nested tuples in test files."""
import re

def fix_nested_tuples(filepath):
    """Fix nested tuple issues in test files."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Fix patterns like (('sum', agg_sum), agg_sum) -> ('sum', agg_sum)
    pattern = r"\(\('(sum|mean|std|min|max|count)', agg_\1\), agg_\1\)"
    content = re.sub(pattern, r"('\1', agg_\1)", content)
    
    # Fix patterns like [('sum', agg_sum), ('mean', agg_mean), ('count', agg_count)]
    # that got double-wrapped
    pattern2 = r"\[\('(sum|mean|std|min|max|count)', agg_\1\), \('(sum|mean|std|min|max|count)', agg_\2\), \('(sum|mean|std|min|max|count)', agg_\3\)\]"
    # This is getting complex, let me just look for the double parentheses

    # Simpler: find (( and fix
    content = content.replace("(('", "('")
    content = re.sub(r"\), agg_(sum|mean|std|min|max|count)\)", r")", content)
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"Fixed {filepath}")

files = [
    'tests/test_group_by.py',
    'tests/test_groupby_apply.py',
    'tests/test_custom_agg.py'
]

for filepath in files:
    fix_nested_tuples(filepath)
