"""
K Frequent Elements

Implement top_k_frequent(items, k) returning a list of the k most frequent items.
- If frequencies tie, any order is acceptable.
- items can be ints or strings.

Example:
  items=[1,1,1,2,2,3], k=2 -> [1,2] (order can vary)
"""

if __name__ == "__main__":
    # Use sets for tie-insensitive checks
    out = top_k_frequent([1,1,1,2,2,3], 2)
    assert set(out) == {1,2}, f"expected {{1,2}}, got {out}"

    out = top_k_frequent(["a","b","a","c","b","a"], 2)
    assert set(out) == {"a","b"}, f"expected {{'a','b'}}, got {out}"

    out = top_k_frequent([], 3)
    assert out == [], f"expected [], got {out}"

    print("EX4 passed")

