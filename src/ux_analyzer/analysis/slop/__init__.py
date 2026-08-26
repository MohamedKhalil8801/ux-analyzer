"""Slop-detection package.

Ports the deterministic 27-rule AI-design-slop fingerprint + 9 copy patterns
and the 0-100 scoring / tier math from ravidsrk/slop-detect
(@slop-detect/core, MIT) so ux-analyzer can score any rendered page with the
same reference behavior. Detection is pure Python over a `VisualSnapshot`
computed-style tree; capture-side JS feeds it extra page metadata (viewport,
document height, scroll position, extracted text context).
"""

from ux_analyzer.analysis.slop.pipeline import analyze_slop
from ux_analyzer.analysis.slop.scoring import DEFINITIONS_VERSION

__all__ = ["analyze_slop", "DEFINITIONS_VERSION"]
