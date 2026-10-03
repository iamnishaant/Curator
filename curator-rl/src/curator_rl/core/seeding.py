"""Named, order-independent random seed streams (Roadmap Part N).

A stream deterministically depends on (master_seed, name) only — never on the
order in which streams are requested.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _name_key(name: str, bits: int = 64) -> int:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=bits // 8).digest()
    return int.from_bytes(digest, "little")


class SeedManager:
    def __init__(self, master_seed: int) -> None:
        self.master_seed = int(master_seed)
        self._rngs: dict[str, np.random.Generator] = {}

    def rng(self, name: str) -> np.random.Generator:
        """Generator for a named stream. Same (master, name) -> same draws."""
        if name not in self._rngs:
            seq = np.random.SeedSequence(entropy=self.master_seed, spawn_key=(_name_key(name),))
            self._rngs[name] = np.random.Generator(np.random.PCG64(seq))
        return self._rngs[name]

    def int_seed(self, name: str, bits: int = 32) -> int:
        if bits <= 0 or bits % 8 != 0 or bits > 512:
            raise ValueError("bits must be a positive multiple of 8 up to 512")
        digest = hashlib.sha256(f"{self.master_seed}:{name}".encode()).digest()
        return int.from_bytes(digest[: bits // 8], "little")
