"""Provider boundary for model-dependent screenshot saliency inference."""

from __future__ import annotations

from typing import Protocol

from ux_analyzer.domain.saliency import (
    SaliencyPredictionRequest,
    SaliencyPredictionSet,
)


class SaliencyProvider(Protocol):
    """Synchronous contract for one screenshot saliency inference request."""

    def predict(self, request: SaliencyPredictionRequest) -> SaliencyPredictionSet: ...


__all__ = ["SaliencyProvider"]
