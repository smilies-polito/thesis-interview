"""
EXERCISE 2 — K-th Largest Element

Implement kth_largest(nums, k) returning the k-th largest element.
- k is 1-based (k=1 returns max)
- Not k-th distinct (duplicates count)

Example:
  nums=[3,2,1,5,6,4], k=2 -> 5
"""

from typing import List

def kth_largest(nums: List[int], k: int) -> int:
    # TODO: implement
    raise NotImplementedError


if __name__ == "__main__":
    tests = [
        (([3,2,1,5,6,4], 2), 5),
        (([3,2,3,1,2,4,5,5,6], 4), 4),
        (([1], 1), 1),
        (([2,2,2], 2), 2),
    ]
    for (nums, k), exp in tests:
        out = kth_largest(nums, k)
        assert out == exp, f"kth_largest({nums}, {k}) => {out}, expected {exp}"
    print("EX2 passed")
