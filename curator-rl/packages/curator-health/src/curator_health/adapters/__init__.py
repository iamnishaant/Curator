"""Framework adapters for curator-health.

Adapters are thin and optional. Import one only if you use that framework; the core package stays
dependency-free. Each adapter preserves the host's return object and never raises into the training
path.
"""

from __future__ import annotations

__all__ = ["trl"]
