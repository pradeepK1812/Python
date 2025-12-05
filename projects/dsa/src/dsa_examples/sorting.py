from typing import List

def merge_sort(arr: List[int]) -> List[int]:
    """Return a sorted copy of arr using merge sort (pure python)."""
    if len(arr) <= 1:
        return arr[:]
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    i = j = 0
    out = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out

def has_duplicate_fast(arr: List[int]) -> bool:
    """Return True if array has any duplicate using set membership."""
    seen = set()
    for x in arr:
        if x in seen:
            return True
        seen.add(x)
    return False
