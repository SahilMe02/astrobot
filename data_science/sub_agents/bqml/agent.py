"""
BigQuery ML specialist agent for the AstroBot Data Science system.

Security model:

    LLM
      ->
    explicit approved BQML tools
      ->
    approved SQL helper
      ->
    BigQuery

No generic arbitrary-SQL execution tool is exposed to the model.
"""

from __future__ import annotations

import os

import google.auth
from dotenv import load_dotenv

from google.adk.agents import Agent

from .prompts import return_instructions_bqml

from .tools import (
    check_bqml_dataset,
    get_bqml_dataset,
    get_customer_segmentation_distribution,
    get_customer_segmentation_predictions,
    get_model_status,
    run_revenue_forecast,
    evaluate_revenue_model,
    explain_revenue_forecast,
    train_revenue_model,
    train_customer_segmentation_model,
)


# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# Model configuration
# ============================================================

BQML_MODEL = os.getenv(
    "BQML_AGENT_MODEL",
    os.getenv(
        "DS_AGENT_MODEL",
        "gemini-2.5-flash",
    ),
)


# ============================================================
# Google Cloud authentication
# ============================================================

credentials, default_project = (
    google.auth.default()
)


if not default_project:
    raise RuntimeError(
        "Google Application Default Credentials did not provide "
        "a default project."
    )


# ============================================================
# Compute configuration
# ============================================================

COMPUTE_PROJECT_ID = os.getenv(
    "GOOGLE_CLOUD_PROJECT",
    default_project,
)

BQML_LOCATION = os.getenv(
    "BQML_LOCATION",
    "US",
)


# ============================================================
# BQML specialist
# ============================================================

root_agent = Agent(
    name="bq_ml_agent",
    model=BQML_MODEL,
    description=(
        "BigQuery ML specialist for approved revenue forecasting "
        "and customer segmentation workflows."
    ),
    instruction=return_instructions_bqml(),
    tools=[
        # ----------------------------------------------------
        # Infrastructure
        # ----------------------------------------------------

        check_bqml_dataset,
        get_bqml_dataset,

        # ----------------------------------------------------
        # Revenue forecasting
        # ----------------------------------------------------

        train_revenue_model,
        run_revenue_forecast,
        evaluate_revenue_model,
        explain_revenue_forecast,

        # ----------------------------------------------------
        # Customer segmentation
        # ----------------------------------------------------

        train_customer_segmentation_model,
        get_customer_segmentation_predictions,
        get_customer_segmentation_distribution,

        # ----------------------------------------------------
        # Model metadata
        # ----------------------------------------------------

        get_model_status,
    ],
)