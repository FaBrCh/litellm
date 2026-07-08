"""Tests for Fireworks AI cache-aware cost calculation.

Regression tests for https://github.com/BerriAI/litellm/issues/31714 and
https://github.com/BerriAI/litellm/issues/25950: cached prompt tokens must be
charged at ``cache_read_input_token_cost``, not at the full input rate.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.llms.fireworks_ai.cost_calculator import cost_per_token
from litellm.types.utils import PromptTokensDetailsWrapper, Usage

GLM_MODEL = "accounts/fireworks/models/glm-5p2"
INPUT_COST = 1.4e-06
CACHE_READ_COST = 1.4e-07
OUTPUT_COST = 4.4e-06


@pytest.fixture(autouse=True)
def register_glm_pricing():
    litellm.register_model(
        {
            f"fireworks_ai/{GLM_MODEL}": {
                "input_cost_per_token": INPUT_COST,
                "output_cost_per_token": OUTPUT_COST,
                "cache_read_input_token_cost": CACHE_READ_COST,
                "litellm_provider": "fireworks_ai",
                "mode": "chat",
            }
        }
    )


def test_cached_tokens_charged_at_cache_read_rate():
    # real-world usage captured from a Fireworks serverless response with a
    # warm prompt cache (prompt_tokens includes the cached portion)
    usage = Usage(
        prompt_tokens=8973,
        completion_tokens=4,
        total_tokens=8977,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=8972),
    )
    prompt_cost, completion_cost = cost_per_token(model=GLM_MODEL, usage=usage)

    expected_prompt = (8973 - 8972) * INPUT_COST + 8972 * CACHE_READ_COST
    assert prompt_cost == pytest.approx(expected_prompt)
    assert completion_cost == pytest.approx(4 * OUTPUT_COST)


def test_uncached_usage_unchanged():
    usage = Usage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100)
    prompt_cost, completion_cost = cost_per_token(model=GLM_MODEL, usage=usage)

    assert prompt_cost == pytest.approx(1000 * INPUT_COST)
    assert completion_cost == pytest.approx(100 * OUTPUT_COST)


def test_unmapped_model_falls_back_to_size_bucket_pricing():
    usage = Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110)
    prompt_cost, completion_cost = cost_per_token(
        model="accounts/fireworks/models/unmapped-model-7b", usage=usage
    )

    assert prompt_cost > 0
    assert completion_cost > 0
