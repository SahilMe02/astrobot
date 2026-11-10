"""BigQuery specialist agent for the AstroBot Data Science system."""

import logging
import os
from typing import Any

import google.auth
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import BaseTool, ToolContext
from google.adk.tools.bigquery import (
    BigQueryCredentialsConfig,
    BigQueryToolset,
)
from google.adk.tools.bigquery.config import BigQueryToolConfig, WriteMode
from google.genai import types

from tenancy import data_sources
from tenancy.context import active_client

from . import tools
from .prompts import return_instructions_bigquery


load_dotenv()


logger = logging.getLogger(__name__)


ADK_BUILTIN_BQ_EXECUTE_SQL_TOOL = "execute_sql"


# ============================================================
# Runtime configuration
# ============================================================

# BigQuery jobs should execute in the user's configured
# application project, NOT in the public data source project.
credentials, adc_project = google.auth.default()

BIGQUERY_COMPUTE_PROJECT = (
    os.getenv("GOOGLE_CLOUD_PROJECT")
    or adc_project
)

if not BIGQUERY_COMPUTE_PROJECT:
    raise RuntimeError(
        "Google Cloud project is not configured. "
        "Set GOOGLE_CLOUD_PROJECT or configure ADC."
    )


# The current public source dataset and the BQML model are in
# the US multi-region. Keep BigQuery job location separate from
# GOOGLE_CLOUD_LOCATION, which may be set to a regional value
# such as us-central1 for Vertex AI.
BIGQUERY_LOCATION = os.getenv(
    "BIGQUERY_LOCATION",
    "US",
)


# ============================================================
# Before-agent callback
# ============================================================

def setup_before_agent_call(
    callback_context: CallbackContext,
) -> None:
    """Load database settings into agent state before execution."""

    # Reload when missing or when cached for a different client.
    settings = callback_context.state.get("database_settings") or {}
    if settings.get("client_id") != active_client() or "tables" not in settings:
        callback_context.state["database_settings"] = (
            tools.get_database_settings()
        )


# ============================================================
# After-tool callback
# ============================================================

def store_results_in_context(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    tool_response: dict[str, Any],
) -> None:
    """Store actual BigQuery execution details in agent state."""

    if tool.name != ADK_BUILTIN_BQ_EXECUTE_SQL_TOOL:
        return

    sql_arg = (
        args.get("query")
        or args.get("sql")
        or ""
    )

    if sql_arg:
        tool_context.state["last_executed_sql"] = sql_arg

    if tool_response.get("status") == "SUCCESS":
        rows = tool_response.get("rows", [])

        tool_context.state["bigquery_query_result"] = rows

        logger.info(
            "BigQuery query executed successfully. Rows returned: %d",
            len(rows),
        )


# ============================================================
# BigQuery tool configuration
# ============================================================

bigquery_tool_filter = [
    ADK_BUILTIN_BQ_EXECUTE_SQL_TOOL,
]


bigquery_credentials_config = BigQueryCredentialsConfig(
    credentials=credentials,
)


bigquery_tool_config = BigQueryToolConfig(
    write_mode=WriteMode.BLOCKED,
    application_name="astrobot-data-science",
    compute_project_id=BIGQUERY_COMPUTE_PROJECT,
    location=BIGQUERY_LOCATION,
)


# ============================================================
# BigQuery toolset
# ============================================================

bigquery_toolset = BigQueryToolset(
    credentials_config=bigquery_credentials_config,
    tool_filter=bigquery_tool_filter,
    bigquery_tool_config=bigquery_tool_config,
)


# ============================================================
# Model configuration
# ============================================================

BIGQUERY_MODEL = os.getenv(
    "DS_BIGQUERY_MODEL",
    os.getenv(
        "DS_AGENT_MODEL",
        "gemini-2.5-flash",
    ),
)


# ============================================================
# BigQuery specialist agent
# ============================================================

bigquery_agent = LlmAgent(
    name="bigquery_agent",
    model=BIGQUERY_MODEL,
    instruction=lambda _ctx: return_instructions_bigquery(
        data_sources.current()
    ),
    tools=[
        tools.bigquery_nl2sql,
        bigquery_toolset,
    ],
    before_agent_callback=setup_before_agent_call,
    # Data policy: every execute_sql call is checked against the active
    # client's table allow-list before it can reach BigQuery.
    before_tool_callback=tools.guard_execute_sql,
    after_tool_callback=store_results_in_context,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.01,
    ),
)


# ADK entrypoint expected by the Data Science subsystem.
root_agent = bigquery_agent
