"""Weight-to-quota conversion (Roadmap D.4) — moved to `core.quotas` in Phase 5
so the L1 scheduler can import it (layering rule B.2); kept as a re-export."""

from curator_rl.core.quotas import largest_remainder_quotas

__all__ = ["largest_remainder_quotas"]
