"""llmevalkit hallucination detection module.

8 metrics for universal hallucination detection across any LLM application.
Every metric works offline (free) or with LLM-as-judge (deeper).

Works with RAG pipelines, AI agents, chatbots, document extraction,
code generation, summarization, and any system that produces text.

Usage:
    from llmevalkit.hallucination import (
        EntityHallucination, NumericHallucination, NegationHallucination,
        FabricatedInfo, ContradictionDetector, SelfConsistency,
        ConfidenceCalibration, InstructionHallucination,
    )

Disclaimer: These metrics help detect potential hallucinations in
LLM outputs. They do not guarantee detection of all hallucinations.
Always verify critical outputs with domain experts.
"""

from llmevalkit.hallucination.entity_hallucination import EntityHallucination
from llmevalkit.hallucination.numeric_hallucination import NumericHallucination
from llmevalkit.hallucination.core_detectors import (
    NegationHallucination, FabricatedInfo, ContradictionDetector,
)
from llmevalkit.hallucination.advanced_detectors import (
    SelfConsistency, ConfidenceCalibration, InstructionHallucination,
)
from llmevalkit.hallucination.extended_detectors import (
    SourceCoverage, TemporalHallucination, CausalHallucination, RankingHallucination,
)

__all__ = [
    "EntityHallucination",
    "NumericHallucination",
    "NegationHallucination",
    "FabricatedInfo",
    "ContradictionDetector",
    "SelfConsistency",
    "ConfidenceCalibration",
    "InstructionHallucination",
    "SourceCoverage",
    "TemporalHallucination",
    "CausalHallucination",
    "RankingHallucination",
]
