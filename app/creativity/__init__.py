"""Creativity wrappers — composes lower-level primitives for brainstorm + reverie use.

The base novelty + lessons-learned + analogy machinery lives in
:mod:`app.self_improvement.novelty` and :mod:`app.companion.lessons_learned`.
This package wraps them for the specific surfaces (brainstorm, reverie)
that need a *combined* assessment.
"""

from app.creativity.analogy_index import (
    AnalogyEntry,
    AnalogyMatch,
    DomainExample,
    add_entry,
    list_all,
    query_analogies,
)
from app.creativity.concept_blend import (
    BlendResult,
    InputSpace,
    blend_concepts,
)
from app.creativity.novelty_wrap import (
    NoveltyVerdict,
    NoveltyWrap,
    assess_brainstorm_idea,
)

__all__ = [
    "AnalogyEntry",
    "AnalogyMatch",
    "BlendResult",
    "DomainExample",
    "InputSpace",
    "NoveltyVerdict",
    "NoveltyWrap",
    "add_entry",
    "assess_brainstorm_idea",
    "blend_concepts",
    "list_all",
    "query_analogies",
]
