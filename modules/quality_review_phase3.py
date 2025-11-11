"""
Quality Review Phase 3: Context Relevance Verification

Verifies that Phase 2 filing context is relevant to Phase 1 bullet/paragraph content.
Detects irrelevant context sentences that don't connect to the actual topic.
"""

import json
import logging
import time
from typing import Dict, List, Optional
import google.generativeai as genai

LOG = logging.getLogger(__name__)

# Gemini verification prompt for context relevance
CONTEXT_RELEVANCE_PROMPT = """You are validating filing context relevance for financial intelligence reports.

This check runs across thousands of companies - keep evaluation straightforward and consistent.

INPUT:
- Content: [bullet text OR paragraph text]
- Context: [filing-derived context sentences]

TASK:
For each sentence in Context, determine: Does this sentence connect to the Content?

═══════════════════════════════════════════════════════════════════
THREE RELEVANCE TESTS
═══════════════════════════════════════════════════════════════════

1. CONNECTION TEST
   Does the sentence:
   ✓ Measure the same thing for the company (market stat → company stat)
   ✓ Explain impact on the company (event → company effect)
   ✗ List unrelated company facts

2. SPECIFICITY TEST
   Is the sentence:
   ✓ Specific to Content's topic(s)
   ✗ Generic company info unrelated to topic

   Note: For paragraphs covering multiple themes, sentence passes if it relates
         to ANY theme mentioned in the paragraph.

3. CONNECTOR TEST
   Can you connect Content to sentence with one transition word?
   ✓ "Content fact → [because/therefore/despite/validates] → Context sentence"
   ✗ Requires multiple logical steps to connect

   Note: For paragraphs with multiple themes, sentence passes if it connects
         to ANY theme mentioned in the Content.

═══════════════════════════════════════════════════════════════════
CONFIDENCE SCORING
═══════════════════════════════════════════════════════════════════

- HIGH (3/3): Definitely relevant → KEEP
- MEDIUM (2/3): Borderline relevant → KEEP
- LOW (1/3): Probably irrelevant → REMOVE
- NONE (0/3): Definitely irrelevant → REMOVE

═══════════════════════════════════════════════════════════════════
SENTENCE SPLITTING
═══════════════════════════════════════════════════════════════════

- Split context on periods followed by spaces
- Preserve semicolons within sentences (don't split on semicolons)
- Each sentence evaluated independently

═══════════════════════════════════════════════════════════════════
CRITICAL RULES
═══════════════════════════════════════════════════════════════════

1. Focus on relevance only - not materiality, not strategic importance
2. For paragraphs: sentence passes Test 2 if it relates to ANY theme mentioned
3. Be consistent across all 5000 companies - same standards
4. Multi-step connections FAIL Test 3
5. Generic stats FAIL Test 2 unless Content is about that specific metric
6. MEDIUM confidence acceptable - keep but monitor
7. If all removed → use escape hatch: "No relevant filing context found for this development"

═══════════════════════════════════════════════════════════════════
JSON OUTPUT STRUCTURE
═══════════════════════════════════════════════════════════════════

Return a JSON object with this exact structure:

{
  "evaluations": [
    {
      "section": "section_name",
      "bullet_id": "id" or null for paragraphs,
      "content": "content text",
      "context_original": "full context text",
      "sentences": [
        {
          "sentence_num": 1,
          "text": "sentence text",
          "test1_result": "PASS" or "FAIL",
          "test1_reason": "brief reason",
          "test2_result": "PASS" or "FAIL",
          "test2_reason": "brief reason",
          "test3_result": "PASS" or "FAIL",
          "test3_reason": "connection explanation",
          "confidence": "HIGH" or "MEDIUM" or "LOW" or "NONE",
          "confidence_score": "X/3",
          "decision": "KEEP" or "REMOVE"
        }
      ],
      "cleaned_context": "reconstructed context with KEEP sentences only",
      "removal_count": integer
    }
  ]
}

If cleaned_context is <20 words, use: "No relevant filing context found for this development"

═══════════════════════════════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════════════════════════════

Example 1: BULLET EVALUATION

Content: "Fannie Mae forecasts 2.9% annual home price appreciation 2025-2029"
Context: "Company targets 5.5-6.0x debt ratio per 10-K; same store NOI growth 1.1% reflects moderation from 2.25% guidance; below-historical appreciation aligns with 2.3% revenue growth guidance"

{
  "evaluations": [{
    "section": "major_developments",
    "bullet_id": "fannie_mae_forecast",
    "content": "Fannie Mae forecasts 2.9% annual home price appreciation 2025-2029",
    "context_original": "Company targets 5.5-6.0x debt ratio per 10-K; same store NOI growth 1.1% reflects moderation from 2.25% guidance; below-historical appreciation aligns with 2.3% revenue growth guidance",
    "sentences": [
      {
        "sentence_num": 1,
        "text": "Company targets 5.5-6.0x debt ratio per 10-K",
        "test1_result": "FAIL",
        "test1_reason": "Debt ratio doesn't measure appreciation",
        "test2_result": "FAIL",
        "test2_reason": "Generic leverage metric",
        "test3_result": "FAIL",
        "test3_reason": "Cannot connect appreciation → debt ratio",
        "confidence": "NONE",
        "confidence_score": "0/3",
        "decision": "REMOVE"
      },
      {
        "sentence_num": 2,
        "text": "same store NOI growth 1.1% reflects moderation from 2.25% guidance",
        "test1_result": "PASS",
        "test1_reason": "Shows company growth moderating like market",
        "test2_result": "PASS",
        "test2_reason": "Specific to growth trajectory",
        "test3_result": "PASS",
        "test3_reason": "Appreciation moderating → consistent with → NOI moderating",
        "confidence": "HIGH",
        "confidence_score": "3/3",
        "decision": "KEEP"
      },
      {
        "sentence_num": 3,
        "text": "below-historical appreciation aligns with 2.3% revenue growth guidance",
        "test1_result": "PASS",
        "test1_reason": "Connects market forecast to company outlook",
        "test2_result": "PASS",
        "test2_reason": "Specific to appreciation/growth",
        "test3_result": "PASS",
        "test3_reason": "2.9% appreciation → aligns with → 2.3% guidance",
        "confidence": "HIGH",
        "confidence_score": "3/3",
        "decision": "KEEP"
      }
    ],
    "cleaned_context": "Same store NOI growth 1.1% reflects moderation from 2.25% guidance; below-historical appreciation aligns with 2.3% revenue growth guidance",
    "removal_count": 1
  }]
}

---

Example 2: COMPLETE REMOVAL

Content: "U.S. industrial absorption reached 60M sqft in Q3"
Context: "Company operates 85,138 homes across 16 markets; Western US 72.1% of revenue; same store rent $2,461 (+2.5% YoY)"

{
  "evaluations": [{
    "section": "competitive_industry_dynamics",
    "bullet_id": "industrial_absorption",
    "content": "U.S. industrial absorption reached 60M sqft in Q3",
    "context_original": "Company operates 85,138 homes across 16 markets; Western US 72.1% of revenue; same store rent $2,461 (+2.5% YoY)",
    "sentences": [
      {
        "sentence_num": 1,
        "text": "Company operates 85,138 homes across 16 markets",
        "test1_result": "FAIL",
        "test1_reason": "Home count doesn't measure absorption",
        "test2_result": "FAIL",
        "test2_reason": "Generic portfolio stat",
        "test3_result": "FAIL",
        "test3_reason": "Cannot connect absorption → home count",
        "confidence": "NONE",
        "confidence_score": "0/3",
        "decision": "REMOVE"
      },
      {
        "sentence_num": 2,
        "text": "Western US 72.1% of revenue",
        "test1_result": "FAIL",
        "test1_reason": "Revenue split doesn't relate to absorption",
        "test2_result": "FAIL",
        "test2_reason": "Generic geographic breakdown",
        "test3_result": "FAIL",
        "test3_reason": "Cannot connect absorption → revenue split",
        "confidence": "NONE",
        "confidence_score": "0/3",
        "decision": "REMOVE"
      },
      {
        "sentence_num": 3,
        "text": "same store rent $2,461 (+2.5% YoY)",
        "test1_result": "FAIL",
        "test1_reason": "Rent level doesn't measure absorption",
        "test2_result": "FAIL",
        "test2_reason": "Generic performance metric",
        "test3_result": "FAIL",
        "test3_reason": "Cannot connect absorption → rent",
        "confidence": "NONE",
        "confidence_score": "0/3",
        "decision": "REMOVE"
      }
    ],
    "cleaned_context": "No relevant filing context found for this development",
    "removal_count": 3
  }]
}
"""


def review_context_relevance(
    ticker: str,
    executive_summary: Dict,
    gemini_api_key: str
) -> Optional[Dict]:
    """
    Review context relevance for executive summary using Gemini.

    Evaluates Phase 2 context sentences against Phase 1 content to identify
    irrelevant filing data that doesn't connect to the actual topic.

    Args:
        ticker: Stock ticker
        executive_summary: Full Phase 1+2+3 merged executive summary JSON
        gemini_api_key: Gemini API key

    Returns:
        {
            "summary": {
                "ticker": str,
                "items_evaluated": int,
                "sentences_evaluated": int,
                "sentences_keep": int,
                "sentences_remove": int,
                "items_with_issues": int,
                "generation_time_ms": int
            },
            "evaluations": [...]  # Detailed review by item
        }
    """
    if not gemini_api_key:
        LOG.error("Gemini API key not configured")
        return None

    try:
        genai.configure(api_key=gemini_api_key)

        # Build list of items to evaluate (dynamic - check for context field)
        items_to_evaluate = []

        sections = executive_summary.get("sections", {})

        for section_name, section_data in sections.items():
            if isinstance(section_data, list):  # Bullet sections
                for bullet in section_data:
                    if isinstance(bullet, dict) and bullet.get('context'):
                        items_to_evaluate.append({
                            "section": section_name,
                            "bullet_id": bullet.get('bullet_id'),
                            "content": bullet.get('content', ''),
                            "context": bullet.get('context', '')
                        })
            elif isinstance(section_data, dict):  # Paragraph sections
                if section_data.get('context'):
                    items_to_evaluate.append({
                        "section": section_name,
                        "bullet_id": None,
                        "content": section_data.get('content', ''),
                        "context": section_data.get('context', '')
                    })

        if not items_to_evaluate:
            LOG.info(f"[{ticker}] No items with context found for Phase 3 review")
            return {
                "summary": {
                    "ticker": ticker,
                    "items_evaluated": 0,
                    "sentences_evaluated": 0,
                    "sentences_keep": 0,
                    "sentences_remove": 0,
                    "items_with_issues": 0,
                    "generation_time_ms": 0
                },
                "evaluations": []
            }

        LOG.info(f"[{ticker}] Building Phase 3 quality review prompt for {len(items_to_evaluate)} items")

        # Build user content with all items
        user_content = "Evaluate the following items:\n\n"

        for i, item in enumerate(items_to_evaluate, 1):
            user_content += f"ITEM {i}:\n"
            user_content += f"Section: {item['section']}\n"
            if item['bullet_id']:
                user_content += f"Bullet ID: {item['bullet_id']}\n"
            user_content += f"Content: {item['content']}\n"
            user_content += f"Context: {item['context']}\n\n"

        # Log token estimate
        char_count = len(CONTEXT_RELEVANCE_PROMPT) + len(user_content)
        token_estimate = char_count // 4
        LOG.info(f"[{ticker}] Phase 3 quality review prompt: {char_count:,} chars (~{token_estimate:,} tokens)")

        # Call Gemini 2.5 Flash
        LOG.info(f"[{ticker}] Calling Gemini 2.5 Flash for Phase 3 quality review")
        model = genai.GenerativeModel('gemini-2.5-flash')

        generation_config = {
            "temperature": 0.0,  # Maximum determinism
            "response_mime_type": "application/json"  # Force JSON output
        }

        start_time = time.time()
        response = model.generate_content(
            CONTEXT_RELEVANCE_PROMPT + "\n\n" + user_content,
            generation_config=generation_config
        )
        generation_time = int((time.time() - start_time) * 1000)  # ms

        # Parse response
        result_text = response.text
        result_json = json.loads(result_text)

        # Calculate summary statistics
        items_evaluated = len(result_json.get("evaluations", []))
        sentences_evaluated = 0
        sentences_keep = 0
        sentences_remove = 0
        items_with_issues = 0

        for evaluation in result_json.get("evaluations", []):
            sentences = evaluation.get("sentences", [])
            sentences_evaluated += len(sentences)

            removal_count = evaluation.get("removal_count", 0)
            if removal_count > 0:
                items_with_issues += 1

            for sentence in sentences:
                decision = sentence.get("decision")
                if decision == "KEEP":
                    sentences_keep += 1
                elif decision == "REMOVE":
                    sentences_remove += 1

        # Build final result
        final_result = {
            "summary": {
                "ticker": ticker,
                "items_evaluated": items_evaluated,
                "sentences_evaluated": sentences_evaluated,
                "sentences_keep": sentences_keep,
                "sentences_remove": sentences_remove,
                "items_with_issues": items_with_issues,
                "generation_time_ms": generation_time
            },
            "evaluations": result_json.get("evaluations", [])
        }

        LOG.info(f"✅ [{ticker}] Phase 3 quality review complete: {items_evaluated} items, "
                f"{sentences_remove}/{sentences_evaluated} sentences flagged for removal")

        return final_result

    except json.JSONDecodeError as e:
        LOG.error(f"[{ticker}] Failed to parse Gemini Phase 3 response as JSON: {e}")
        return None
    except Exception as e:
        LOG.error(f"[{ticker}] Phase 3 quality review failed: {e}")
        import traceback
        LOG.error(traceback.format_exc())
        return None
