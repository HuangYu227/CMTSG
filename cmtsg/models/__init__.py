from __future__ import annotations

__all__ = ["CMTSGModel", "CausalSemanticGrounding"]


def __getattr__(name: str):
    if name == "CMTSGModel":
        from cmtsg.models.cmtsg import CMTSGModel

        return CMTSGModel
    if name == "CausalSemanticGrounding":
        from cmtsg.models.grounding import CausalSemanticGrounding

        return CausalSemanticGrounding
    raise AttributeError(name)
