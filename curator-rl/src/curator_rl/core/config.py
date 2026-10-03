"""Validated configuration system.

Scientific hyperparameters have NO Python defaults: a config file (or
override) must supply them, otherwise validation fails. Unknown keys fail.
Everything is extra="forbid", so the config hash covers the whole experiment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class CuratorConfigError(Exception):
    """Raised for any user-facing configuration problem."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentCfg(_StrictModel):
    name: str
    tier: Literal[0, "1R", 1, 2]
    method: str
    seed: int = Field(ge=0)


class BudgetCfg(_StrictModel):
    total_usd: float = Field(gt=0)
    usd_per_gpu_hour: float = Field(gt=0)
    n_gpus: int = Field(ge=1)


class SchedulerCfg(_StrictModel):
    gamma: float = Field(gt=0, le=1)
    exploration_coef: float = Field(ge=0)
    tau: float = Field(gt=0)
    epsilon: float = Field(ge=0, lt=1)
    warmup_rounds: int = Field(ge=0)
    score_norm: Literal["none", "zscore", "rank"]
    cost_exponent: float = Field(ge=0)
    status_control: Literal["off", "soft", "hard"]
    pull_unit: Literal["round", "prompt"]


class RichnessCfg(_StrictModel):
    mode: Literal["mixed", "band", "variance"]
    band_lo: float = Field(ge=0, le=1)
    band_hi: float = Field(ge=0, le=1)
    variance_min: float = Field(gt=0)

    @field_validator("band_hi")
    @classmethod
    def _band_ordered(cls, hi: float, info):  # noqa: ANN001
        lo = info.data.get("band_lo")
        if lo is not None and hi < lo:
            raise ValueError("richness.band_hi must be >= richness.band_lo")
        return hi


class StatusCfg(_StrictModel):
    n_min: int = Field(gt=0)
    r_min: int = Field(gt=0)
    p_sat: float = Field(ge=0, le=1)
    p_hard: float = Field(ge=0, le=1)
    sr_hard: float = Field(ge=0, le=1)
    z_up: float = Field(gt=0)
    z_neg: float = Field(gt=0)
    hysteresis_p: float = Field(ge=0, le=1)
    hysteresis_sr: float = Field(ge=0, le=1)
    dwell_min: int = Field(ge=1)
    consecutive_rounds: int = Field(ge=1)


class PriorCfg(_StrictModel):
    alpha0: float = Field(gt=0)
    beta0: float = Field(gt=0)


class SignalsCfg(_StrictModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    lam: float = Field(alias="lambda", gt=0, le=1)
    lambda_fast: float = Field(gt=0, le=1)
    lambda_slow: float = Field(gt=0, le=1)
    lp_method: Literal["lp_a", "lp_b", "lp_c"]
    lp_use: Literal["signed", "positive", "abs"]
    richness: RichnessCfg
    prior: PriorCfg
    status: StatusCfg


class ProxyCfg(_StrictModel):
    alpha: float = Field(ge=0)
    beta: float = Field(ge=0)
    clip_l: float = Field(gt=0)


class CalibCfg(_StrictModel):
    interval_rounds: int = Field(ge=1)
    target: Literal["domain", "global"]
    k_min: int = Field(ge=1)
    trust_region: float = Field(gt=0, le=1)
    z_mis: float = Field(gt=0)
    mismatch_windows: int = Field(ge=1)
    mismatch_clear_windows: int = Field(ge=1)


class CostCfg(_StrictModel):
    count_cpu_verifier_as_gpu_time: bool
    warmup_steps: int = Field(ge=0)


class TokenLimitsCfg(_StrictModel):
    max_prompt_tokens: int = Field(gt=0)
    max_completion_tokens: int = Field(gt=0)
    verifier_timeout_s: float = Field(gt=0)
    prior_usd_per_prompt: float = Field(gt=0)


class Gsm8kEnvCfg(TokenLimitsCfg):
    pass


class CountdownEnvCfg(TokenLimitsCfg):
    n_numbers: int = Field(ge=2, le=5)  # le=5 bounds the reachable-target search
    numbers_min: int = Field(ge=1)
    numbers_max: int = Field(ge=2)
    target_min: int = Field(ge=1)
    target_max: int = Field(ge=2)
    use_all_numbers: bool

    @field_validator("numbers_max")
    @classmethod
    def _numbers_ordered(cls, hi: int, info):  # noqa: ANN001
        lo = info.data.get("numbers_min")
        if lo is not None and hi < lo:
            raise ValueError("countdown.numbers_max must be >= countdown.numbers_min")
        return hi

    @field_validator("target_max")
    @classmethod
    def _target_ordered(cls, hi: int, info):  # noqa: ANN001
        lo = info.data.get("target_min")
        if lo is not None and hi < lo:
            raise ValueError("countdown.target_max must be >= countdown.target_min")
        return hi


class NoisyEnvCfg(TokenLimitsCfg):
    mode: Literal["random", "flip"]
    q: float = Field(gt=0, lt=1)      # Bernoulli(q) success rate in 'random' mode
    flip_p: float = Field(gt=0, lt=1)  # flip probability in 'flip' mode


class DataEnvsCfg(_StrictModel):
    gsm8k: Gsm8kEnvCfg
    countdown: CountdownEnvCfg
    noisy: NoisyEnvCfg


class DataCfg(_StrictModel):
    root: str
    split_salt: str
    calib_size: int = Field(ge=1)
    dev_size: int = Field(ge=1)
    envs: DataEnvsCfg


class PathsCfg(_StrictModel):
    runs_root: str


class LoggingCfg(_StrictModel):
    fsync: bool
    schema_version: int = Field(ge=1)


class WarmupCfg(_StrictModel):
    total_rounds: int = Field(ge=0)


class RootConfig(_StrictModel):
    experiment: ExperimentCfg
    budget: BudgetCfg
    scheduler: SchedulerCfg
    signals: SignalsCfg
    proxy: ProxyCfg
    calib: CalibCfg
    cost: CostCfg
    data: DataCfg
    paths: PathsCfg
    logging_cfg: LoggingCfg = Field(alias="logging")
    steps_per_round: int = Field(ge=1)
    prompts_per_step: int = Field(ge=1)
    group_size: int = Field(ge=1)
    warmup: WarmupCfg

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


def _set_by_path(payload: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node: Any = payload
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise CuratorConfigError(
                f"override path '{dotted}' does not exist in the config (missing '{part}')"
            )
        node = node[part]
    if not isinstance(node, dict):
        raise CuratorConfigError(f"override path '{dotted}' traverses a non-mapping")
    node[parts[-1]] = value


def _apply_overrides(payload: dict, overrides: list[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise CuratorConfigError(f"override '{override}' must be of the form a.b=c (missing '=')")
        dotted, raw = override.split("=", 1)
        dotted = dotted.strip()
        if not dotted:
            raise CuratorConfigError(f"override '{override}' has an empty key")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise CuratorConfigError(f"override '{override}' value is not valid YAML: {exc}") from exc
        _set_by_path(payload, dotted, value)


def _deep_merge(base: dict, child: dict) -> dict:
    out = dict(base)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_payload(path: Path) -> dict:
    """Load a YAML file, resolving `include: [files]` relative to the file, child wins."""
    path = Path(path)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CuratorConfigError(f"config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise CuratorConfigError(f"config file {path} is not valid YAML: {exc}") from exc
    if not isinstance(payload, dict):
        raise CuratorConfigError(f"config file {path} must contain a YAML mapping")
    include = payload.pop("include", None)
    if include is not None:
        if isinstance(include, str):
            include = [include]
        if not isinstance(include, list) or not all(isinstance(i, str) for i in include):
            raise CuratorConfigError(f"'include' in {path} must be a path or a list of paths")
        base: dict = {}
        for inc in include:
            base = _deep_merge(base, _load_payload(path.parent / inc))
        payload = _deep_merge(base, payload)
    return payload


def config_dict_to_canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def config_hash(cfg: RootConfig) -> str:
    """SHA-256 (64 hex chars) of the canonical (sorted-key) JSON form of the config."""
    return hashlib.sha256(config_dict_to_canonical(cfg.model_dump(by_alias=True, mode="json")).encode("utf-8")).hexdigest()


def dump_config(cfg: RootConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.model_dump(by_alias=True, mode="json"), sort_keys=False), encoding="utf-8")


def load_config(path: Path, overrides: list[str] | tuple[str, ...] = ()) -> RootConfig:
    """Load a YAML config (with include support), apply 'a.b=c' overrides, and validate."""
    payload = _load_payload(Path(path))
    _apply_overrides(payload, list(overrides))
    try:
        return RootConfig.model_validate(payload)
    except ValidationError as exc:
        raise CuratorConfigError(str(exc)) from exc
