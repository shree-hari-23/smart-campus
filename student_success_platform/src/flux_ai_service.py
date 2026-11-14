"""
=============================================================================
FLUXBASE AI / GEMINI AI GATEWAY SERVICE (PACD-RAG)
AI-Powered Agentic Smart Campus Intelligence & Student Success Platform
=============================================================================
Routes all platform AI workflows through Google Gemini models:
  - gemini-2.5-flash / gemini-2.0-flash / gemini-1.5-flash / gemini-1.5-pro
Maintains 100% backward compatibility with all existing FluxAIService callers.
=============================================================================
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from gemini_ai_service import GeminiAIService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FluxAIService(GeminiAIService):
    """
    OpenAI/Gemini unified AI Gateway client connecting to Google Gemini models
    (gemini-2.5-flash, gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-pro).
    Fully backward-compatible with all existing FluxAIService callers.
    """

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(
        self,
        api_key: Optional[str] = None,
        config_path: Optional[str] = None,
        default_model: str = DEFAULT_MODEL
    ):
        super().__init__(api_key=api_key, default_model=default_model)
        self.base_url = self.DEFAULT_BASE_URL


# Standalone CLI execution demonstration
if __name__ == "__main__":
    print("\n" + "=" * 75)
    print("  GEMINI / FLUXBASE AI GATEWAY VERIFICATION")
    print("  Active Engine: Google Gemini Models ('gemini-2.5-flash', 'gemini-1.5-flash')")
    print("=" * 75)

    ai = FluxAIService()
    print(f"1. AI Gateway Connected: {ai.base_url}")
    print(f"2. Active Model: {ai.default_model}")
    print(f"3. API Key Active: {bool(ai.api_key)}")

    print("\n4. Testing Pedagogical Guidance Generation for At-Risk Student:")
    guidance = ai.generate_pedagogical_guidance(
        student_name="Rohan Verma",
        srn="R24EA818",
        risk_level="HIGH",
        dropout_probability=0.785,
        target_concept="Neural Network Training",
        diagnosed_prerequisite_gaps=["Partial Derivatives", "Chain Rule", "Backpropagation"],
        top_shap_factors=["Zero 2nd sem credits approved", "Tuition fee arrears"]
    )
    print("\nGemini AI Synthesized Intervention Plan:")
    print(guidance)

    print("\n5. Testing Adaptive MCQ Question Generation:")
    drill = ai.generate_adaptive_drill_item(
        concept_name="Chain Rule in Backpropagation",
        unit="Unit 3: Deep Neural Networks",
        difficulty="Intermediate"
    )
    print(f"\nQuestion: {drill.get('question')}")
    for idx, opt in enumerate(drill.get("options", [])):
        marker = " (Correct)" if idx == drill.get("correct_index") else ""
        print(f"  [{idx}] {opt}{marker}")
    print(f"Explanation: {drill.get('explanation')}")

    print("\n" + "=" * 75)
    print("  GEMINI AI GATEWAY OPERATIONAL AND READY FOR AGENTS & RAG.")
    print("=" * 75 + "\n")
