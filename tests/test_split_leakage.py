"""Ensures the hash-bucket split function itself is leakage-free and deterministic."""
import sys
sys.path.insert(0, ".")
from src.data.split import bucket


def test_bucket_deterministic():
    h = "abc123def4560000"
    assert bucket(h) == bucket(h)


def test_bucket_in_range():
    for h in ["0000000000000000", "ffffffffffffffff", "abc123def4560000"]:
        b = bucket(h)
        assert 0 <= b < 100
