"""Test range compression for filtered indices."""
import numpy as np
import sys
sys.path.insert(0, '/Users/jwilliam/git/vibehdf5')

from vibehdf5.hdf5_viewer import indices_to_ranges, ranges_to_indices


def test_ranges():
    """Test the range compression and decompression functions."""

    # Test case 1: Consecutive range
    indices1 = [1, 2, 3, 4, 5, 10]
    ranges1 = indices_to_ranges(indices1)
    print(f"Test 1: {indices1} -> {ranges1}")
    assert ranges1 == ['1-5', 10], f"Expected ['1-5', 10], got {ranges1}"
    recovered1 = ranges_to_indices(ranges1)
    print(f"  Recovered: {recovered1.tolist()}")
    assert np.array_equal(recovered1, indices1), "Mismatch after recovery"

    # Test case 2: All isolated indices
    indices2 = [1, 3, 5, 7, 9]
    ranges2 = indices_to_ranges(indices2)
    print(f"\nTest 2: {indices2} -> {ranges2}")
    assert ranges2 == [1, 3, 5, 7, 9], f"Expected [1, 3, 5, 7, 9], got {ranges2}"
    recovered2 = ranges_to_indices(ranges2)
    print(f"  Recovered: {recovered2.tolist()}")
    assert np.array_equal(recovered2, indices2), "Mismatch after recovery"

    # Test case 3: Multiple ranges
    indices3 = [1, 2, 3, 10, 11, 12, 20]
    ranges3 = indices_to_ranges(indices3)
    print(f"\nTest 3: {indices3} -> {ranges3}")
    assert ranges3 == ['1-3', '10-12', 20], f"Expected ['1-3', '10-12', 20], got {ranges3}"
    recovered3 = ranges_to_indices(ranges3)
    print(f"  Recovered: {recovered3.tolist()}")
    assert np.array_equal(recovered3, indices3), "Mismatch after recovery"

    # Test case 4: Single element
    indices4 = [42]
    ranges4 = indices_to_ranges(indices4)
    print(f"\nTest 4: {indices4} -> {ranges4}")
    assert ranges4 == [42], f"Expected [42], got {ranges4}"
    recovered4 = ranges_to_indices(ranges4)
    print(f"  Recovered: {recovered4.tolist()}")
    assert np.array_equal(recovered4, indices4), "Mismatch after recovery"

    # Test case 5: Empty list
    indices5 = []
    ranges5 = indices_to_ranges(indices5)
    print(f"\nTest 5: {indices5} -> {ranges5}")
    assert ranges5 == [], f"Expected [], got {ranges5}"
    recovered5 = ranges_to_indices(ranges5)
    print(f"  Recovered: {recovered5.tolist()}")
    assert len(recovered5) == 0, "Expected empty array"

    # Test case 6: Large consecutive range
    indices6 = list(range(0, 10000))
    ranges6 = indices_to_ranges(indices6)
    print(f"\nTest 6: range(0, 10000) -> {ranges6}")
    assert ranges6 == ['0-9999'], f"Expected ['0-9999'], got {ranges6}"
    recovered6 = ranges_to_indices(ranges6)
    print(f"  Recovered length: {len(recovered6)}")
    assert np.array_equal(recovered6, indices6), "Mismatch after recovery"

    # Test case 7: NumPy array input
    indices7 = np.array([5, 6, 7, 15, 16, 17, 18, 25])
    ranges7 = indices_to_ranges(indices7)
    print(f"\nTest 7: np.array({indices7.tolist()}) -> {ranges7}")
    assert ranges7 == ['5-7', '15-18', 25], f"Expected ['5-7', '15-18', 25], got {ranges7}"
    recovered7 = ranges_to_indices(ranges7)
    print(f"  Recovered: {recovered7.tolist()}")
    assert np.array_equal(recovered7, indices7), "Mismatch after recovery"

    print("\n✅ All tests passed!")

    # Demonstrate space savings
    print("\n--- Space Savings Demonstration ---")
    import json

    test_cases = [
        ("Small consecutive", list(range(10))),
        ("Large consecutive", list(range(10000))),
        ("Mixed pattern", [1,2,3,4,5] + [10,11,12] + [20,21,22,23,24,25] + [100,200,300]),
        ("Sparse", list(range(0, 1000, 10))),
    ]

    for name, indices in test_cases:
        ranges = indices_to_ranges(indices)
        original_size = len(json.dumps(indices))
        compressed_size = len(json.dumps(ranges))
        savings = (1 - compressed_size / original_size) * 100
        print(f"\n{name}:")
        print(f"  Original indices: {len(indices)} items, {original_size} bytes")
        print(f"  Compact ranges: {len(ranges)} items, {compressed_size} bytes")
        print(f"  Space savings: {savings:.1f}%")


if __name__ == "__main__":
    test_ranges()
