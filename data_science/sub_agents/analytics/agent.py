"""Analytics specialist agent.

Supports two modes:

1. json
   Generates a structured chart specification.
   The parent Data Science tools can render the specification.

2. code_interpreter
   Uses Vertex AI Code Executor for advanced chart generation.
"""

import logging
import os
from functools import cached_property

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models import Gemini
from google.genai import Client, types

from .prompts import (
    return_instructions_analytics,
    return_instructions_code_interpreter,
)


load_dotenv()

logger = logging.getLogger(__name__)


class GlobalGemini(Gemini):
    """Gemini model wrapper that can use the global Vertex AI location."""

    @cached_property
    def api_client(self) -> Client:
        return Client(
            vertexai=True,
            location="global",
        )


def _build_analytics_code_executor():
    """Create the optional Vertex AI Code Executor.

    The executor is only enabled when a real extension resource name
    is provided through CODE_INTERPRETER_EXTENSION_NAME.
    """

    extension_name = os.getenv(
        "CODE_INTERPRETER_EXTENSION_NAME",
        "",
    ).strip()

    if not extension_name:
        logger.info(
            "CODE_INTERPRETER_EXTENSION_NAME is not configured. "
            "Analytics will use JSON mode."
        )
        return None

    try:
        from google.adk.code_executors import VertexAiCodeExecutor

        return VertexAiCodeExecutor(
            resource_name=extension_name,
            optimize_data_file=True,
            stateful=True,
        )

    except Exception as exc:
        logger.warning(
            "Could not initialize VertexAiCodeExecutor: %r. "
            "Falling back to JSON mode.",
            exc,
        )
        return None


ANALYTICS_MODE = os.getenv(
    "ANALYTICS_MODE",
    "json",
).strip().lower()

CODE_EXECUTOR_AVAILABLE = bool(
    os.getenv(
        "CODE_INTERPRETER_EXTENSION_NAME",
        "",
    ).strip()
)

USE_CODE_INTERPRETER = (
    ANALYTICS_MODE == "code_interpreter"
    and CODE_EXECUTOR_AVAILABLE
)


if USE_CODE_INTERPRETER:

    code_executor = _build_analytics_code_executor()

    if code_executor is not None:

        analytics_agent = Agent(
            model=os.getenv(
                "ANALYTICS_AGENT_MODEL",
                os.getenv(
                    "DS_AGENT_MODEL",
                    "gemini-2.5-flash",
                ),
            ),
            name="analytics_agent",
            instruction=return_instructions_code_interpreter(),
            code_executor=code_executor,
            generate_content_config=types.GenerateContentConfig(
                temperature=0.0,
            ),
        )

    else:
        logger.warning(
            "Code interpreter was requested but could not be initialized. "
            "Using JSON analytics mode."
        )

        analytics_agent = Agent(
            model=GlobalGemini(
                model=os.getenv(
                    "ANALYTICS_AGENT_MODEL",
                    os.getenv(
                        "DS_AGENT_MODEL",
                        "gemini-2.5-flash",
                    ),
                )
            ),
            name="analytics_agent",
            instruction=return_instructions_analytics(),
            generate_content_config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )

else:

    analytics_agent = Agent(
        model=GlobalGemini(
            model=os.getenv(
                "ANALYTICS_AGENT_MODEL",
                os.getenv(
                    "DS_AGENT_MODEL",
                    "gemini-2.5-flash",
                ),
            )
        ),
        name="analytics_agent",
        instruction=return_instructions_analytics(),
        generate_content_config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )


# ADK's simple folder-based runner can discover this as the entry point.
root_agent = analytics_agent