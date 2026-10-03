"""Seeding tests: reproducibility and stream independence."""

import pytest

from curator_rl.core.seeding import SeedManager


def test_seed_streams_reproducible_and_independent():
    a = SeedManager(123)
    b = SeedManager(123)
    xa = [a.rng("data_order").uniform() for _ in range(5)]
    xb = [b.rng("data_order").uniform() for _ in range(5)]
    assert xa == xb

    other = [SeedManager(123).rng("sampler").uniform() for _ in range(5)]
    assert other != xa


def test_seed_order_independence():
    # requesting a then b equals requesting b then a
    ab: list[float] = []
    m1 = SeedManager(7)
    ab.append(m1.rng("a").uniform())
    ab.append(m1.rng("b").uniform())

    ba: list[float] = []
    m2 = SeedManager(7)
    ba.append(m2.rng("b").uniform())
    ba.append(m2.rng("a").uniform())

    assert ba == ab[::-1]


def test_int_seed_deterministic_and_order_independent():
    s1 = SeedManager(42).int_seed("bootstrap")
    s2 = SeedManager(42).int_seed("bootstrap")
    m = SeedManager(42)
    m.rng("x").uniform()
    s3 = m.int_seed("bootstrap")
    assert s1 == s2 == s3
    assert SeedManager(42).int_seed("other") != s1
    with pytest.raises(ValueError):
        SeedManager(42).int_seed("bad", bits=31)


def test_rng_streams_are_statistically_independent():
    m = SeedManager(0)
    draws = [m.rng(name).uniform(-1e9, 1e9) for name in ["a", "b", "c", "d"]]
    assert len(set(draws)) == 4  # four distinct streams, distinct values
    assert draws[0] != draws[3]
