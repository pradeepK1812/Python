
from dsa_examples.sorting import merge_sort, has_duplicate_fast

def test_merge_sort_basic():
    arr = [5,3,1,4,2]
    assert merge_sort(arr) == [1,2,3,4,5]
    # ensure original not mutated
    assert arr == [5,3,1,4,2]

def test_merge_sort_empty():
    assert merge_sort([]) == []

def test_has_duplicate_fast():
    assert has_duplicate_fast([1,2,3,2]) is True
    assert has_duplicate_fast([1,2,3]) is False

