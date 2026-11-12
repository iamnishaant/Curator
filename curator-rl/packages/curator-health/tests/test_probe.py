"""Pre-flight probe: a leaky checker is flagged, a strict one is not, and errors never propagate."""

from __future__ import annotations

from curator_health import ProbePrompt, probe_all, probe_arm, probes

PROMPT = ProbePrompt(
    prompt_id="p1",
    messages=({"role": "user", "content": "What is 6 times 7?"},),
    reference="42",
)


def strict(prompt, response):
    return 1.0 if f"\\boxed{{{prompt.reference}}}" in response else 0.0


def leaky(prompt, response):
    return 1.0 if prompt.reference in response else 0.0


def raising(prompt, response):
    raise RuntimeError("boom")


def test_probe_list_covers_the_junk_families():
    names = {name for name, _ in probes("gsm8k", PROMPT)}
    assert {"empty", "echo_prompt", "empty_box", "wrong_box", "number_spray"} <= names
    assert "empty_code" in {n for n, _ in probes("mbpp", PROMPT)}


def test_a_strict_verifier_is_not_flagged():
    report = probe_arm("gsm8k", strict, [PROMPT])
    assert report["junk_reward_rate"] == 0.0
    assert not report["flagged"]


def test_a_leaky_checker_is_flagged():
    report = probe_arm("gsm8k", leaky, [PROMPT])
    assert report["flagged"]
    assert report["probes"]["number_spray"] == 1.0


def test_a_raising_verifier_is_fail_open():
    report = probe_arm("gsm8k", raising, [PROMPT])
    assert report["errors"] > 0
    assert not report["flagged"]


def test_probe_all_reports_every_arm_and_lists_the_flagged_ones():
    specs = {"gsm8k": (strict, [PROMPT]), "noisy": (leaky, [PROMPT])}
    report = probe_all(specs)
    assert report["flagged"] == ["noisy"]
    assert set(report["arms"]) == {"gsm8k", "noisy"}


def test_countdown_probe_uses_the_target_literal():
    prompt = ProbePrompt(prompt_id="c1", messages=({"role": "user", "content": "make 24"},),
                         meta={"target": 24, "numbers": [2, 3, 4, 6]})
    names = {name for name, _ in probes("countdown", prompt)}
    assert "target_literal" in names
