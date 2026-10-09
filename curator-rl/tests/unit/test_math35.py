"""Unit tests for the MATH answer extraction and equivalence checker (Phase E, no data needed)."""

import pytest

from curator_rl.envs.math35 import answers_equal, extract_boxed, normalize_answer, parse_level


def test_extract_boxed_handles_nested_braces_and_takes_the_last():
    assert extract_boxed(r"so \boxed{\frac{1}{2}} and finally \boxed{\frac{3}{4}}") == r"\frac{3}{4}"
    assert extract_boxed(r"\boxed{x^{2}+1}") == "x^{2}+1"
    assert extract_boxed("no answer here") is None
    assert extract_boxed(r"\boxed{unterminated") is None
    assert extract_boxed(r"first \boxed{7} then broken \boxed{") == "7"      # falls back to the last COMPLETE one
    assert extract_boxed(r"\boxed {5}") == "5"


def test_parse_level():
    assert parse_level("Level 3") == 3
    assert parse_level("Level ?") is None
    assert parse_level("") is None


@pytest.mark.parametrize("a,b", [
    ("0.5", r"\frac{1}{2}"), (r"\dfrac{1}{2}", "1/2"), (r"\frac12", "0.5"), ("1,000", "1000"),
    ("x = 5", "5"), (r"5\text{ cm}", "5"), (r"90^\circ", "90"), (r"\$12.50", "12.5"), ("25\\%", "25"),
    (r"\left(1,2\right)", "(1,2)"), ("7.", "7"), (r"\text{yes}", "yes"), (r"-\frac{3}{4}", "-0.75"),
])
def test_equivalent_spellings_are_accepted(a, b):
    assert answers_equal(a, b)


@pytest.mark.parametrize("a,b", [
    (r"\frac{1}{2}", r"\frac{1}{3}"), ("2", "3"), ("10", "1,0"), (r"\sqrt{2}", r"\sqrt{3}"),
    ("(1,2)", "(2,1)"), ("5", "-5"), ("yes", "no"), (r"\frac{1}{2}", "2"), ("12", "21"),
])
def test_different_values_are_never_accepted(a, b):
    assert not answers_equal(a, b)


def test_normalisation_is_idempotent():
    for s in (r"\dfrac{3}{4}", "x = 12", r"5\text{ meters}", "1,234.50"):
        once = normalize_answer(s)
        assert normalize_answer(once) == once


def test_empty_and_garbage_never_match_a_real_answer():
    for g in ("", " ", "abc", r"\boxed{}", "None"):
        assert not answers_equal(g, "5")
        assert not answers_equal(g, r"\frac{1}{2}")
