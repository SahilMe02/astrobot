"""ADK evaluations of the Orchestrator (google.adk.evaluation.AgentEvaluator).

  isolation/  deterministic guardrail cases; no model call, so they run in CI
  routing/    live end-to-end cases judged by an LLM against rubrics; they need
              Vertex AI, BigQuery, Gmail and FRED, so they only run when
              ASTROBOT_LIVE_EVALS=1 (on a developer machine with gcloud auth)

Same files work with the ADK CLI, e.g.
  adk eval orchestrator orchestrator/evals/routing/routing.test.json \
      --config_file_path orchestrator/evals/routing/test_config.json --print_detailed_results
"""

import asyncio
import os
from pathlib import Path

import pytest
from google.adk.evaluation.agent_evaluator import AgentEvaluator
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_set import EvalSet

EVALS = Path(__file__).parent / "evals"


@pytest.mark.parametrize("name", ["isolation", "routing"])
def test_eval_files_are_valid(name):
    eval_set = EvalSet.model_validate_json((EVALS / name / f"{name}.test.json").read_text())
    EvalConfig.model_validate_json((EVALS / name / "test_config.json").read_text())
    for case in eval_set.eval_cases:
        # Cases must run as a user on the access list (or a deliberate intruder).
        assert case.session_input and case.session_input.app_name == "orchestrator"


def test_isolation_evals():
    asyncio.run(AgentEvaluator.evaluate(
        "orchestrator", str(EVALS / "isolation" / "isolation.test.json"), num_runs=1
    ))


@pytest.mark.skipif(os.getenv("ASTROBOT_LIVE_EVALS") != "1",
                    reason="live eval: set ASTROBOT_LIVE_EVALS=1 (needs GCP credentials)")
def test_routing_evals():
    asyncio.run(AgentEvaluator.evaluate(
        "orchestrator", str(EVALS / "routing" / "routing.test.json"), num_runs=1
    ))
