"""
Code utilities for OpenPACEvolve.
"""

import re
from typing import List, Optional, Tuple


def extract_evolve_blocks(code: str) -> List[Tuple[int, int, str]]:
    """
    Extract EVOLVE-BLOCK regions from code.
    
    Returns list of (start_line, end_line, block_content) tuples.
    """
    lines = code.split("\n")
    blocks = []
    in_block = False
    start_line = 0
    block_lines = []
    
    for i, line in enumerate(lines):
        if "EVOLVE-BLOCK-START" in line:
            in_block = True
            start_line = i
            block_lines = []
        elif "EVOLVE-BLOCK-END" in line:
            if in_block:
                blocks.append((start_line, i, "\n".join(block_lines)))
            in_block = False
        elif in_block:
            block_lines.append(line)
    
    return blocks


def merge_code_changes(original: str, evolved_block: str, block_idx: int = 0) -> str:
    """
    Merge evolved code block back into original code.
    
    Args:
        original: Original code with EVOLVE-BLOCK markers.
        evolved_block: New code to insert.
        block_idx: Which block to replace (if multiple).
        
    Returns:
        Merged code.
    """
    blocks = extract_evolve_blocks(original)
    
    if block_idx >= len(blocks):
        return original
    
    start, end, _ = blocks[block_idx]
    lines = original.split("\n")
    
    # Build merged code
    merged_lines = lines[:start + 1]  # Include start marker
    merged_lines.extend(evolved_block.split("\n"))
    merged_lines.extend(lines[end:])  # Include end marker and after
    
    return "\n".join(merged_lines)


def normalize_code(code: str) -> str:
    """
    Normalize code for comparison.
    
    - Removes comments
    - Normalizes whitespace
    """
    # Remove single-line comments
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    
    # Normalize whitespace
    code = re.sub(r'\s+', ' ', code)
    
    return code.strip()


def code_similarity(code1: str, code2: str) -> float:
    """
    Calculate similarity between two code snippets.
    
    Returns value between 0 and 1.
    """
    norm1 = normalize_code(code1)
    norm2 = normalize_code(code2)
    
    if not norm1 or not norm2:
        return 0.0
    
    # Simple character-level similarity
    set1 = set(norm1.split())
    set2 = set(norm2.split())
    
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0
