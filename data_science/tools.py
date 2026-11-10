"""Tool wrappers for the Data Science root agent.

Current capabilities:
- BigQuery specialist delegation
- BigQuery retry handling
- BigQuery result persistence in session state
- Analytics specialist delegation
- Chart-spec validation
- Matplotlib chart rendering
- ADK artifact persistence
- BQML specialist delegation
"""

import hashlib
import io
import json
import logging
import math
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from google.adk.tools import ToolContext
from google.adk.tools.agent_tool import AgentTool
from google.genai import types


load_dotenv()

_logger = logging.getLogger(__name__)

_MAX_BQ_RETRIES = 2
_MAX_ANALYTICS_ROWS = 5000


# ============================================================
# Generic helpers
# ============================================================

def _sanitize_json_value(value: Any) -> Any:
    """Convert Python values into JSON-safe values."""

    if isinstance(value, dict):
        return {
            str(key): _sanitize_json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _sanitize_json_value(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            _sanitize_json_value(item)
            for item in value
        ]

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

    return value


def _to_json_text(value: Any) -> str:
    """Serialize a value safely for model context."""

    return json.dumps(
        _sanitize_json_value(value),
        default=str,
        ensure_ascii=False,
    )


# ============================================================
# BigQuery specialist
# ============================================================

_CHART_WORDS = re.compile(r"\b(chart|graph|plot|visuali[sz]e|visual)\b", re.I)
_REFERENCE_WORDS = re.compile(r"\b(that|this|it|above|previous|those|these|same)\b", re.I)
_NEW_DATA_WORDS = re.compile(
    r"\b(by|per|for|last|this year|month|week|day|quarter|year|since|between|compare|vs)\b",
    re.I,
)


def _is_chart_followup(user_text: str) -> bool:
    """User's own message asks to chart the previous result, not new data."""
    text = str(user_text or "")
    return bool(
        _CHART_WORDS.search(text)
        and _REFERENCE_WORDS.search(text)
        and not _NEW_DATA_WORDS.search(text)
    )


async def call_bigquery_agent(
    query: str,
    tool_context: ToolContext,
) -> str:
    """Delegate data retrieval to the BigQuery specialist."""

    if not query or not query.strip():
        return "BigQuery request is empty."

    # "Chart that": reuse the stored result instead of re-querying, so the
    # chart shows exactly the numbers the user already saw.
    stored = tool_context.state.get("last_bq_result")
    if stored and _is_chart_followup(
        tool_context.state.get("_current_user_request", "")
    ):
        _logger.info("Chart follow-up: reusing stored BigQuery result.")
        tool_context.state["last_visualization_result"] = stored
        tool_context.state["last_visualization_source"] = "bigquery"
        return str(stored)

    query_hash = hashlib.sha256(
        query.strip().encode("utf-8")
    ).hexdigest()[:12]

    previous_hash = tool_context.state.get(
        "_last_bq_hash",
        "",
    )

    previous_result = tool_context.state.get(
        "last_bq_result",
        "",
    )

    # Avoid repeating the exact same request unnecessarily.
    if (
        query_hash == previous_hash
        and previous_result
    ):
        _logger.info(
            "Duplicate BigQuery request detected: %s",
            query_hash,
        )

        # Re-expose the existing BigQuery result as the
        # active result for downstream visualization.
        tool_context.state[
            "last_visualization_result"
        ] = previous_result

        tool_context.state[
            "last_visualization_source"
        ] = "bigquery"

        return str(previous_result)

    tool_context.state[
        "_last_bq_hash"
    ] = query_hash

    from .sub_agents.bigquery.agent import bigquery_agent

    agent_tool = AgentTool(
        agent=bigquery_agent,
    )

    last_error: str | None = None

    for attempt in range(
        _MAX_BQ_RETRIES + 1
    ):
        tool_context.state[
            "_bq_retry_count"
        ] = attempt

        actual_request = query

        if attempt > 0 and last_error:
            actual_request = (
                f"{query}\n\n"
                "The previous attempt failed.\n"
                f"Previous error: {last_error[:500]}\n"
                "Regenerate a simpler read-only query."
            )

        try:
            _logger.info(
                "Calling BigQuery specialist. attempt=%d",
                attempt + 1,
            )

            result = await agent_tool.run_async(
                args={
                    "request": actual_request,
                },
                tool_context=tool_context,
            )

            result_text = str(result)

            lowered = result_text.lower()

            possible_errors = (
                "syntax error",
                "unrecognized name",
                "table not found",
                "does not exist",
                "invalid query",
                "sql error",
            )

            looks_like_error = any(
                error_text in lowered
                for error_text in possible_errors
            )

            if looks_like_error:
                last_error = result_text[:500]

                _logger.warning(
                    "Possible BigQuery failure on attempt %d: %s",
                    attempt + 1,
                    last_error[:250],
                )

                if attempt < _MAX_BQ_RETRIES:
                    continue

            # This is the authoritative result that downstream
            # specialists can consume.
            tool_context.state[
                "last_bq_result"
            ] = result_text

            # Also expose the result under a clearer key for
            # downstream analytics consumers.
            tool_context.state[
                "bigquery_query_result"
            ] = result_text

            # Canonical result for downstream visualization.
            tool_context.state[
                "last_visualization_result"
            ] = result_text

            # Identify the producer of the current visualization
            # data.
            tool_context.state[
                "last_visualization_source"
            ] = "bigquery"

            tool_context.state[
                "_bq_retry_count"
            ] = 0

            return result_text

        except Exception as exc:
            last_error = str(exc)[:500]

            _logger.exception(
                "BigQuery specialist failed on attempt %d.",
                attempt + 1,
            )

            if attempt >= _MAX_BQ_RETRIES:
                break

    tool_context.state[
        "_bq_retry_count"
    ] = 0

    failure_message = (
        "BigQuery request failed after "
        f"{_MAX_BQ_RETRIES + 1} attempts."
    )

    if last_error:
        failure_message += (
            f" Last error: {last_error}"
        )

    tool_context.state[
        "last_bq_result"
    ] = failure_message

    tool_context.state[
        "bigquery_query_result"
    ] = failure_message

    return failure_message


# ============================================================
# Chart specification validation
# ============================================================

def _validate_chart_spec(
    spec: dict[str, Any],
) -> tuple[bool, str]:
    """Validate a chart specification before rendering."""

    if not isinstance(spec, dict):
        return (
            False,
            "Chart specification must be a JSON object.",
        )

    allowed_chart_types = {
        "bar",
        "stacked_bar",
        "line",
        "scatter",
        "cluster_scatter",
        "heatmap",
        "pie",
        "funnel",
        "waterfall",
        "sankey",
    }

    chart_type = spec.get(
        "chart_type"
    )

    if chart_type not in allowed_chart_types:
        return (
            False,
            f"Unsupported chart type: {chart_type!r}",
        )

    data = spec.get("data")

    if not isinstance(data, dict):
        return (
            False,
            "Chart data must be a JSON object.",
        )

    # --------------------------------------------------------
    # BAR / PIE / FUNNEL / WATERFALL
    # --------------------------------------------------------

    if chart_type in {
        "bar",
        "pie",
        "funnel",
        "waterfall",
    }:
        categories = data.get(
            "categories"
        )

        values = data.get(
            "values"
        )

        if not isinstance(categories, list):
            return (
                False,
                "Chart categories must be a list.",
            )

        series = data.get("series")
        if (
            chart_type == "bar"
            and values is None
            and isinstance(series, dict)
            and series
        ):
            # Multi-metric bar: one panel per series (see renderer).
            for name, series_values in series.items():
                if not isinstance(series_values, list):
                    return False, f"Series {name!r} must be a list."
                if len(series_values) != len(categories):
                    return False, f"Series {name!r} length does not match categories."
            values = categories

        if not isinstance(values, list):
            return (
                False,
                "Chart values must be a list.",
            )

        if not categories:
            return (
                False,
                "Chart categories cannot be empty.",
            )

        if len(categories) != len(values):
            return (
                False,
                "Categories and values have different lengths.",
            )

    # --------------------------------------------------------
    # STACKED BAR
    # --------------------------------------------------------

    elif chart_type == "stacked_bar":
        categories = data.get(
            "categories"
        )

        series = data.get(
            "series"
        )

        if not isinstance(categories, list):
            return (
                False,
                "Stacked bar categories must be a list.",
            )

        if not categories:
            return (
                False,
                "Stacked bar categories cannot be empty.",
            )

        if not isinstance(series, dict):
            return (
                False,
                "Stacked bar requires a series object.",
            )

        if not series:
            return (
                False,
                "Stacked bar series cannot be empty.",
            )

        for name, values in series.items():
            if not isinstance(values, list):
                return (
                    False,
                    f"Series {name!r} must be a list.",
                )

            if len(values) != len(categories):
                return (
                    False,
                    f"Series {name!r} length does not "
                    "match categories.",
                )

    # --------------------------------------------------------
    # LINE
    # --------------------------------------------------------

    elif chart_type == "line":
        categories = data.get(
            "categories"
        )

        if not isinstance(categories, list):
            return (
                False,
                "Line categories must be a list.",
            )

        if not categories:
            return (
                False,
                "Line chart categories are missing.",
            )

        series = data.get(
            "series"
        )

        if series is not None:
            if not isinstance(series, dict):
                return (
                    False,
                    "Line series must be an object.",
                )

            if not series:
                return (
                    False,
                    "Line series cannot be empty.",
                )

            for name, values in series.items():
                if not isinstance(values, list):
                    return (
                        False,
                        f"Series {name!r} must be a list.",
                    )

                if len(values) != len(categories):
                    return (
                        False,
                        f"Series {name!r} length does not "
                        "match categories.",
                    )

        else:
            values = data.get(
                "values"
            )

            if not isinstance(values, list):
                return (
                    False,
                    "Line chart values are missing.",
                )

            if len(values) != len(categories):
                return (
                    False,
                    "Line categories and values have "
                    "different lengths.",
                )

    # --------------------------------------------------------
    # SCATTER
    # --------------------------------------------------------

    elif chart_type in {
        "scatter",
        "cluster_scatter",
    }:
        x_values = data.get("x")
        y_values = data.get("y")

        if not isinstance(x_values, list):
            return (
                False,
                "Scatter x values must be a list.",
            )

        if not isinstance(y_values, list):
            return (
                False,
                "Scatter y values must be a list.",
            )

        if len(x_values) != len(y_values):
            return (
                False,
                "Scatter x and y lengths do not match.",
            )

        if len(x_values) < 2:
            return (
                False,
                "Scatter needs at least two points.",
            )

        if chart_type == "cluster_scatter":
            clusters = data.get(
                "clusters"
            )

            if not isinstance(clusters, list):
                return (
                    False,
                    "Cluster scatter requires clusters.",
                )

            if len(clusters) != len(x_values):
                return (
                    False,
                    "Clusters length does not match x/y.",
                )

    # --------------------------------------------------------
    # HEATMAP
    # --------------------------------------------------------

    elif chart_type == "heatmap":
        matrix = data.get(
            "matrix"
        )

        if matrix is not None:
            labels = data.get(
                "labels"
            )

            if not isinstance(matrix, list):
                return (
                    False,
                    "Heatmap matrix must be a list.",
                )

            if not matrix:
                return (
                    False,
                    "Heatmap matrix is empty.",
                )

            if not isinstance(labels, list):
                return (
                    False,
                    "Heatmap labels are missing.",
                )

            matrix_size = len(matrix)

            if len(labels) != matrix_size:
                return (
                    False,
                    "Heatmap labels do not match matrix size.",
                )

            for row in matrix:
                if (
                    not isinstance(row, list)
                    or len(row) != matrix_size
                ):
                    return (
                        False,
                        "Heatmap matrix must be square.",
                    )

        else:
            series = data.get(
                "series"
            )

            if not isinstance(series, dict):
                return (
                    False,
                    "Heatmap requires matrix or series.",
                )

            if len(series) < 2:
                return (
                    False,
                    "Heatmap requires at least two series.",
                )

            series_lengths = set()

            for values in series.values():
                if not isinstance(values, list):
                    return (
                        False,
                        "Heatmap series values must be lists.",
                    )

                series_lengths.add(
                    len(values)
                )

            if len(series_lengths) != 1:
                return (
                    False,
                    "Heatmap series have different lengths.",
                )

    # --------------------------------------------------------
    # SANKEY
    # --------------------------------------------------------

    elif chart_type == "sankey":
        flows = data.get(
            "flows"
        )

        if not isinstance(flows, list):
            return (
                False,
                "Sankey requires a flows list.",
            )

        if not flows:
            return (
                False,
                "Sankey flows cannot be empty.",
            )

        for flow in flows:
            if not isinstance(flow, dict):
                return (
                    False,
                    "Each Sankey flow must be an object.",
                )

            if "from" not in flow:
                return (
                    False,
                    "Sankey flow is missing 'from'.",
                )

            if "to" not in flow:
                return (
                    False,
                    "Sankey flow is missing 'to'.",
                )

            if "value" not in flow:
                return (
                    False,
                    "Sankey flow is missing 'value'.",
                )

    return True, "ok"


# ============================================================
# Chart renderer
# ============================================================

def _render_multi_metric_bar(spec: dict[str, Any]) -> bytes:
    """Bar chart with one panel per metric, each on its own scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    categories = spec["data"]["categories"]
    series = spec["data"]["series"]
    fig, axes = plt.subplots(
        1, len(series), figsize=(7 * len(series), 7), squeeze=False
    )
    for ax, (name, values) in zip(axes[0], series.items()):
        bars = ax.bar(categories, [float(v) for v in values])
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set_title(str(name))
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle(spec.get("title", "Analytics Chart"))
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    return buffer.getvalue()


def _render_chart_to_bytes(
    spec: dict[str, Any],
) -> bytes:
    """Render a validated chart specification to PNG bytes."""

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np

    chart_type = spec[
        "chart_type"
    ]

    title = spec.get(
        "title",
        "Analytics Chart",
    )

    x_label = spec.get(
        "x_label",
        "",
    )

    y_label = spec.get(
        "y_label",
        "",
    )

    data = spec[
        "data"
    ]

    if (
        chart_type == "bar"
        and "values" not in data
        and isinstance(data.get("series"), dict)
    ):
        return _render_multi_metric_bar(spec)

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    # --------------------------------------------------------
    # BAR
    # --------------------------------------------------------

    if chart_type == "bar":
        categories = data[
            "categories"
        ]

        values = [
            float(value)
            for value in data[
                "values"
            ]
        ]

        bars = ax.bar(
            categories,
            values,
        )

        ax.bar_label(
            bars,
            fmt="%.2f",
            padding=3,
        )

        ax.tick_params(
            axis="x",
            rotation=45,
        )

    # --------------------------------------------------------
    # STACKED BAR
    # --------------------------------------------------------

    elif chart_type == "stacked_bar":
        categories = data[
            "categories"
        ]

        series = data[
            "series"
        ]

        bottom = np.zeros(
            len(categories)
        )

        for name, values in series.items():
            numeric_values = np.asarray(
                values,
                dtype=float,
            )

            ax.bar(
                categories,
                numeric_values,
                bottom=bottom,
                label=name,
            )

            bottom += numeric_values

        ax.legend()

        ax.tick_params(
            axis="x",
            rotation=45,
        )

    # --------------------------------------------------------
    # LINE
    # --------------------------------------------------------

    elif chart_type == "line":
        categories = data[
            "categories"
        ]

        series = data.get(
            "series"
        )

        if series:
            for name, values in series.items():
                ax.plot(
                    categories,
                    values,
                    marker="o",
                    label=name,
                )

            ax.legend()

        else:
            ax.plot(
                categories,
                data["values"],
                marker="o",
            )

        ax.tick_params(
            axis="x",
            rotation=45,
        )

    # --------------------------------------------------------
    # SCATTER
    # --------------------------------------------------------

    elif chart_type == "scatter":
        ax.scatter(
            data["x"],
            data["y"],
        )

    # --------------------------------------------------------
    # CLUSTER SCATTER
    # --------------------------------------------------------

    elif chart_type == "cluster_scatter":
        ax.scatter(
            data["x"],
            data["y"],
            c=data["clusters"],
        )

    # --------------------------------------------------------
    # PIE
    # --------------------------------------------------------

    elif chart_type == "pie":
        values = [
            float(value)
            for value in data[
                "values"
            ]
        ]

        ax.pie(
            values,
            labels=data[
                "categories"
            ],
            autopct="%1.1f%%",
        )

        ax.axis(
            "equal"
        )

    # --------------------------------------------------------
    # HEATMAP
    # --------------------------------------------------------

    elif chart_type == "heatmap":
        matrix = data.get(
            "matrix"
        )

        labels = data.get(
            "labels"
        )

        if matrix is None:
            series = data[
                "series"
            ]

            labels = list(
                series.keys()
            )

            matrix_values = np.asarray(
                [
                    series[name]
                    for name in labels
                ],
                dtype=float,
            )

            matrix = np.corrcoef(
                matrix_values
            )

        matrix_array = np.asarray(
            matrix,
            dtype=float,
        )

        heatmap = ax.imshow(
            matrix_array,
            aspect="auto",
            vmin=-1,
            vmax=1,
        )

        ax.set_xticks(
            range(
                len(labels)
            )
        )

        ax.set_yticks(
            range(
                len(labels)
            )
        )

        ax.set_xticklabels(
            labels,
            rotation=45,
            ha="right",
        )

        ax.set_yticklabels(
            labels
        )

        fig.colorbar(
            heatmap,
            ax=ax,
        )

    # --------------------------------------------------------
    # FUNNEL
    # --------------------------------------------------------

    elif chart_type == "funnel":
        categories = data[
            "categories"
        ]

        values = np.asarray(
            data["values"],
            dtype=float,
        )

        positions = np.arange(
            len(categories)
        )

        max_value = (
            float(np.max(values))
            if len(values)
            else 1.0
        )

        if max_value != 0:
            widths = (
                values / max_value
            )
        else:
            widths = values

        ax.barh(
            positions,
            widths,
        )

        ax.set_yticks(
            positions
        )

        ax.set_yticklabels(
            categories
        )

        ax.invert_yaxis()

    # --------------------------------------------------------
    # WATERFALL
    # --------------------------------------------------------

    elif chart_type == "waterfall":
        categories = data[
            "categories"
        ]

        values = [
            float(value)
            for value in data[
                "values"
            ]
        ]

        if not values:
            raise ValueError(
                "Waterfall contains no values."
            )

        bottoms = []
        heights = []

        cumulative = values[0]

        bottoms.append(
            0.0
        )

        heights.append(
            values[0]
        )

        for value in values[1:-1]:
            if value >= 0:
                bottoms.append(
                    cumulative
                )
                heights.append(
                    value
                )
            else:
                cumulative += value
                bottoms.append(
                    cumulative
                )
                heights.append(
                    abs(value)
                )
                continue

            cumulative += value

        if len(values) > 1:
            bottoms.append(
                0.0
            )
            heights.append(
                cumulative
            )

        ax.bar(
            categories,
            heights,
            bottom=bottoms,
        )

        ax.axhline(
            0,
            linewidth=1,
        )

        ax.tick_params(
            axis="x",
            rotation=45,
        )

    # --------------------------------------------------------
    # SANKEY
    # --------------------------------------------------------

    elif chart_type == "sankey":
        flows = data[
            "flows"
        ]

        source_names = []
        target_names = []

        for flow in flows:
            source = str(
                flow["from"]
            )

            target = str(
                flow["to"]
            )

            if source not in source_names:
                source_names.append(
                    source
                )

            if target not in target_names:
                target_names.append(
                    target
                )

        source_y = {
            name: index
            for index, name
            in enumerate(source_names)
        }

        target_y = {
            name: index
            for index, name
            in enumerate(target_names)
        }

        max_value = max(
            float(flow["value"])
            for flow in flows
        )

        for flow in flows:
            source = str(
                flow["from"]
            )

            target = str(
                flow["to"]
            )

            value = float(
                flow["value"]
            )

            linewidth = max(
                1.0,
                8.0 * value / max_value
                if max_value
                else 1.0,
            )

            ax.plot(
                [0, 1],
                [
                    source_y[source],
                    target_y[target],
                ],
                linewidth=linewidth,
                alpha=0.5,
            )

        ax.set_xlim(
            -0.1,
            1.1,
        )

        ax.set_xticks(
            [0, 1]
        )

        ax.set_xticklabels(
            [
                "Source",
                "Target",
            ]
        )

        ax.set_yticks([])

    else:
        raise ValueError(
            f"Unsupported renderer chart type: "
            f"{chart_type!r}"
        )

    ax.set_title(
        title
    )

    if x_label:
        ax.set_xlabel(
            x_label
        )

    if y_label:
        ax.set_ylabel(
            y_label
        )

    fig.tight_layout()

    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=120,
        bbox_inches="tight",
    )

    buffer.seek(0)

    plt.close(
        fig
    )

    return buffer.read()

def _is_chart_request(request: str) -> bool:
    """Return True when the request asks for visualization output."""
    normalized = re.sub(
        r"\s+",
        " ",
        request.lower(),
    ).strip()

    chart_terms = (
        "bar chart",
        "line chart",
        "chart",
        "plot",
        "graph",
        "visualize",
        "visualization",
    )

    return any(
        term in normalized
        for term in chart_terms
    )


def _is_customer_segmentation_distribution_request(
    request: str,
) -> bool:
    """Return True when the request explicitly asks for segment distribution/counts."""
    normalized = re.sub(
        r"\s+",
        " ",
        request.lower(),
    ).strip()

    segment_terms = (
        "customer segment",
        "customer segments",
        "segmentation",
        "segment",
        "segments",
        "cluster",
        "clusters",
        "centroid",
        "centroids",
    )

    distribution_terms = (
        "distribution",
        "distribution of customers",
        "customer distribution",
        "distribution across",
        "count of customers",
        "customer count",
        "customer counts",
        "number of customers",
        "how many customers",
        "segment count",
        "segment counts",
        "segment distribution",
        "cluster count",
        "cluster counts",
    )

    has_segment_context = any(
        term in normalized
        for term in segment_terms
    )

    has_distribution_intent = any(
        term in normalized
        for term in distribution_terms
    )

    return has_segment_context and has_distribution_intent


async def _run_customer_segmentation_distribution(
    tool_context: ToolContext,
) -> str:
    """Run the approved full-population customer segment distribution workflow."""
    from .sub_agents.bqml.tools import (
        _get_client,
        get_customer_segmentation_distribution_sql,
    )

    sql = get_customer_segmentation_distribution_sql()

    _logger.info(
        "Running deterministic customer segmentation distribution workflow."
    )

    client = _get_client()
    query_job = client.query(
        sql,
        location="US",
    )

    rows = [
        dict(row)
        for row in query_job.result()
    ]

    normalized_rows: list[dict[str, Any]] = []

    for row in rows:
        centroid_id = row.get("CENTROID_ID")
        customer_count = row.get("customer_count")

        normalized_rows.append(
            {
                "CENTROID_ID": (
                    int(centroid_id)
                    if centroid_id is not None
                    else None
                ),
                "customer_count": (
                    int(customer_count)
                    if customer_count is not None
                    else None
                ),
            }
        )

    normalized_rows.sort(
        key=lambda item: (
            item["CENTROID_ID"]
            if item["CENTROID_ID"] is not None
            else 999999
        )
    )

    if not normalized_rows:
        raise RuntimeError(
            "Customer segmentation distribution returned no rows."
        )

    if any(
        row["CENTROID_ID"] is None
        or row["customer_count"] is None
        for row in normalized_rows
    ):
        raise RuntimeError(
            "Customer segmentation distribution returned an invalid aggregate result."
        )

    total_customers = sum(
        row["customer_count"]
        for row in normalized_rows
    )

    if total_customers <= 0:
        raise RuntimeError(
            "Customer segmentation distribution returned a non-positive customer total."
        )

    result_text = _to_json_text(
        normalized_rows
    )

    tool_context.state[
        "last_bqml_result"
    ] = result_text

    tool_context.state[
        "last_visualization_result"
    ] = normalized_rows

    tool_context.state[
        "last_visualization_source"
    ] = "bqml"

    tool_context.state[
        "_bqml_workflow"
    ] = "customer_segmentation_distribution"

    tool_context.state[
        "last_bqml_distribution_sql"
    ] = sql

    _logger.info(
        "Customer segment distribution returned %d rows and %d customers.",
        len(normalized_rows),
        total_customers,
    )

    return (
        "Customer segmentation distribution generated from "
        "the approved full-population BQML workflow. "
        f"Total customers: {total_customers}. "
        f"Aggregate rows: {result_text}"
    )


def _is_explainable_forecast_request(request: str) -> bool:
    """Return True when the request targets an explainable BQML forecast."""
    normalized = re.sub(
        r"\s+",
        " ",
        request.lower(),
    ).strip()

    return (
        "forecast" in normalized
        and "explainable" in normalized
    )


def _forecast_label() -> str:
    """What the active client's forecast measures (e.g. Revenue, Conversions)."""
    from tenancy import data_sources

    try:
        forecast = data_sources.current().forecast
    except data_sources.DataSourceError:
        forecast = None
    return (forecast.label if forecast else "value").title()


def _extract_requested_forecast_horizon(request: str) -> int | None:
    """Extract an explicit day-based forecast horizon from a request."""
    match = re.search(
        r"\b(\d+)\s*[- ]?\s*day(?:s)?\b",
        request,
        re.IGNORECASE,
    )

    if not match:
        return None

    horizon = int(match.group(1))

    if horizon < 1 or horizon > 365:
        return None

    return horizon


def _build_bqml_explainable_forecast_chart_spec(
    result: Any,
    request: str,
) -> dict[str, Any] | None:
    """Build a deterministic line-chart spec from actual BQML forecast rows."""

    requested_horizon = _extract_requested_forecast_horizon(
        request
    )

    if requested_horizon is None:
        return None

    rows: list[dict[str, Any]] = []

    # --------------------------------------------------------
    # Preferred path: the BQML result is already structured.
    # --------------------------------------------------------
    if isinstance(result, list):
        for item in result:
            if not isinstance(item, dict):
                continue

            forecast_timestamp = item.get(
                "time_series_timestamp"
            ) or item.get(
                "forecast_timestamp"
            )

            forecast_value = item.get(
                "time_series_data"
            ) or item.get(
                "forecast_value"
            )

            lower_bound = item.get(
                "prediction_interval_lower_bound"
            )

            upper_bound = item.get(
                "prediction_interval_upper_bound"
            )

            time_series_type = item.get(
                "time_series_type"
            )

            if (
                forecast_timestamp is None
                or forecast_value is None
                or lower_bound is None
                or upper_bound is None
            ):
                continue

            if (
                time_series_type is not None
                and str(time_series_type).lower() != "forecast"
            ):
                continue

            rows.append(
                {
                    "date": str(forecast_timestamp)[:10],
                    "forecast": float(forecast_value),
                    "lower": float(lower_bound),
                    "upper": float(upper_bound),
                }
            )

    # --------------------------------------------------------
    # Fallback path: the BQML specialist returned formatted
    # forecast text such as:
    # 2026-09-30: Forecast Value: ..., Lower Bound: ...,
    # Upper Bound: ...
    # --------------------------------------------------------
    if not rows and isinstance(result, str):
        forecast_pattern = re.compile(
            r"(?P<date>\d{4}-\d{2}-\d{2})"
            r"[\s\S]{0,180}?Forecast Value:\s*"
            r"(?P<forecast>[-+]?(?:\d+(?:\.\d*)?|\.\d+))"
            r"[\s\S]{0,120}?Lower Bound:\s*"
            r"(?P<lower>[-+]?(?:\d+(?:\.\d*)?|\.\d+))"
            r"[\s\S]{0,120}?Upper Bound:\s*"
            r"(?P<upper>[-+]?(?:\d+(?:\.\d*)?|\.\d+))",
            re.IGNORECASE,
        )

        for match in forecast_pattern.finditer(result):
            rows.append(
                {
                    "date": match.group("date"),
                    "forecast": float(match.group("forecast")),
                    "lower": float(match.group("lower")),
                    "upper": float(match.group("upper")),
                }
            )

    if not rows:
        return None

    rows.sort(
        key=lambda item: item["date"]
    )

    # Never let stale or expanded result text produce more
    # points than the requested forecast horizon.
    rows = rows[:requested_horizon]

    if len(rows) != requested_horizon:
        return None

    return {
        "chart_type": "line",
        "title": (
            f"{requested_horizon}-day Explainable {_forecast_label()} Forecast"
        ),
        "x_label": "Date",
        "y_label": _forecast_label(),
        "data": {
            "categories": [
                row["date"]
                for row in rows
            ],
            "series": {
                "Forecast": [
                    row["forecast"]
                    for row in rows
                ],
                "Lower Bound": [
                    row["lower"]
                    for row in rows
                ],
                "Upper Bound": [
                    row["upper"]
                    for row in rows
                ],
            },
        },
    }


def _build_customer_segmentation_distribution_chart_spec(
    result: Any,
) -> dict[str, Any] | None:
    """Build a deterministic chart spec from aggregate segment counts."""
    if not isinstance(result, list):
        return None

    rows: list[dict[str, Any]] = []

    for item in result:
        if not isinstance(item, dict):
            return None

        if (
            "CENTROID_ID" not in item
            or "customer_count" not in item
        ):
            return None

        rows.append(item)

    if not rows:
        return None

    rows.sort(
        key=lambda item: int(
            item["CENTROID_ID"]
        )
    )

    categories = [
        str(
            int(item["CENTROID_ID"])
        )
        for item in rows
    ]

    values = [
        int(item["customer_count"])
        for item in rows
    ]

    return {
        "chart_type": "bar",
        "title": "Customer Distribution by Segment",
        "x_label": "Segment ID",
        "y_label": "Number of Customers",
        "data": {
            "categories": categories,
            "values": values,
        },
    }


# ============================================================
# Analytics specialist
# ============================================================

async def call_analytics_agent(
    analysis_request: str,
    tool_context: ToolContext,
) -> str:
    """Delegate visualization work to the Analytics specialist."""

    if not analysis_request or not analysis_request.strip():
        return "Analytics request is empty."

    from .sub_agents.analytics.agent import analytics_agent

    # --------------------------------------------------------
    # Use the canonical result produced by the most recent
    # data-producing specialist.
    #
    # This supports both:
    #   BigQuery -> Analytics
    #   BQML     -> Analytics
    #
    # without making Analytics depend on a specific producer.
    # --------------------------------------------------------

    visualization_result = tool_context.state.get(
        "last_visualization_result"
    )

    visualization_source = tool_context.state.get(
        "last_visualization_source"
    )

    # --------------------------------------------------------
    # Backward-compatible fallbacks for existing sessions or
    # workflows that have not populated the canonical key.
    # --------------------------------------------------------

    if not visualization_result:
        visualization_result = (
            tool_context.state.get(
                "bigquery_query_result"
            )
            or tool_context.state.get(
                "last_bq_result"
            )
            or tool_context.state.get(
                "last_bqml_result"
            )
        )

        if visualization_result:
            if (
                tool_context.state.get(
                    "last_bqml_result"
                )
                == visualization_result
            ):
                visualization_source = "bqml"
            else:
                visualization_source = "bigquery"

    if not visualization_result:
        return (
            "Analytics request cannot be completed because "
            "no data-producing specialist result is available."
        )

    # Avoid pushing very large result sets into the model.
    if (
        isinstance(
            visualization_result,
            list,
        )
        and len(visualization_result) > _MAX_ANALYTICS_ROWS
    ):
        return (
            "Analytics request cannot be completed because "
            "the result is too large. "
            "Request an aggregated result first."
        )

    result_text = _to_json_text(
        visualization_result
    )

    analytics_prompt = f"""
Create the requested visualization using ONLY the actual
data-producing specialist result supplied below.

Data source:
{visualization_source or "unknown"}

Visualization request:
{analysis_request}

Actual result:
{result_text}

Rules:
1. Use only values present in the supplied result.
2. Never invent or estimate values.
3. Never create missing categories.
4. Never modify the supplied numerical values.
5. If the data source is BQML, treat forecast values as
   actual model output.
6. For chronological forecast data, use a line chart when
   the user requests a line chart.
7. Return only the chart JSON specification.
8. Do not ask for information that is already present.
"""

    agent_tool = AgentTool(
        agent=analytics_agent,
    )

    try:
        _logger.info(
            "Calling Analytics specialist."
        )

        result = await agent_tool.run_async(
            args={
                "request": analytics_prompt,
            },
            tool_context=tool_context,
        )

    except Exception as exc:
        _logger.exception(
            "Analytics specialist failed."
        )

        return (
            "Analytics generation failed: "
            f"{exc}"
        )

    raw_text = str(
        result
    ).strip()

    # Defensive cleanup in case the model adds fences.
    cleaned_text = (
        raw_text
        .replace(
            "```json",
            "",
        )
        .replace(
            "```",
            "",
        )
        .strip()
    )

    # Extract the first JSON object if there is additional text.
    match = re.search(
        r"\{[\s\S]*\}",
        cleaned_text,
    )

    if not match:
        return (
            "Chart generation failed: "
            "Analytics Agent did not return a JSON object."
        )

    try:
        spec = json.loads(
            match.group(0)
        )

    except json.JSONDecodeError as exc:
        _logger.warning(
            "Analytics JSON parsing failed: %s",
            exc,
        )

        return (
            "Chart generation failed: "
            f"invalid JSON ({exc})."
        )

    is_valid, validation_message = (
        _validate_chart_spec(
            spec
        )
    )

    if not is_valid:
        _logger.warning(
            "Invalid chart specification: %s",
            validation_message,
        )

        return (
            "Chart generation failed: "
            f"{validation_message}"
        )

    # --------------------------------------------------------
    # Deterministic guardrail for customer-segment distribution.
    #
    # Analytics still generates the chart, but its numeric
    # values are validated against the authoritative aggregate
    # BQML result so sample-based counts cannot slip through.
    # --------------------------------------------------------

    if (
        tool_context.state.get(
            "_bqml_workflow"
        )
        == "customer_segmentation_distribution"
    ):
        expected_spec = (
            _build_customer_segmentation_distribution_chart_spec(
                visualization_result
            )
        )

        if expected_spec is None:
            return (
                "Chart generation failed: "
                "the customer segmentation distribution result "
                "is not in the expected aggregate format."
            )

        actual_data = spec.get(
            "data",
            {},
        )

        expected_data = expected_spec["data"]

        actual_categories = [
            str(value)
            for value in actual_data.get(
                "categories",
                [],
            )
        ]

        expected_categories = [
            str(value)
            for value in expected_data["categories"]
        ]

        actual_values = [
            float(value)
            for value in actual_data.get(
                "values",
                [],
            )
        ]

        expected_values = [
            float(value)
            for value in expected_data["values"]
        ]

        if (
            spec.get("chart_type") != "bar"
            or actual_categories != expected_categories
            or actual_values != expected_values
        ):
            _logger.warning(
                "Analytics chart data did not match the "
                "authoritative BQML segment distribution. "
                "Applying deterministic chart-data guardrail."
            )

            spec = expected_spec

    # --------------------------------------------------------
    # Deterministic guardrail for explainable BQML forecasts.
    #
    # The BQML specialist is the source of truth. When the user
    # requests an explainable forecast, rebuild the chart directly
    # from the actual returned forecast rows so stale session data
    # or an over-broad Analytics interpretation cannot expand a
    # 7-day request into 30 points (or any other horizon).
    # --------------------------------------------------------

    if (
        visualization_source == "bqml"
        and (
            _is_explainable_forecast_request(
                analysis_request
            )
            or tool_context.state.get(
                "_bqml_workflow"
            )
            == "revenue_explain_forecast"
        )
        and _is_chart_request(analysis_request)
    ):
        explainable_spec = (
            _build_bqml_explainable_forecast_chart_spec(
                visualization_result,
                analysis_request,
            )
        )

        if explainable_spec is None:
            return (
                "Chart generation failed: "
                "the BQML explainable forecast result did not "
                "contain the requested forecast rows."
            )

        spec = explainable_spec

    try:
        png_bytes = _render_chart_to_bytes(
            spec
        )

    except Exception as exc:
        _logger.exception(
            "Chart rendering failed."
        )

        return (
            "Chart rendering failed: "
            f"{exc}"
        )

    chart_hash = hashlib.sha256(
        _to_json_text(
            spec
        ).encode(
            "utf-8"
        )
    ).hexdigest()[:10]

    filename = (
        f"chart_{int(time.time())}_"
        f"{chart_hash}.png"
    )

    artifact = types.Part.from_bytes(
        data=png_bytes,
        mime_type="image/png",
    )

    try:
        version = await tool_context.save_artifact(
            filename,
            artifact,
        )

        _logger.info(
            "Chart artifact saved: %s version=%s bytes=%d",
            filename,
            version,
            len(png_bytes),
        )

    except Exception as exc:
        _logger.exception(
            "Chart artifact save failed."
        )

        return (
            "Chart was rendered successfully, "
            "but artifact saving failed: "
            f"{exc}"
        )

    tool_context.state[
        "last_chart_spec"
    ] = spec

    tool_context.state[
        "last_chart_artifact"
    ] = filename

    tool_context.state[
        "_chart_steps_done"
    ] = (
        tool_context.state.get(
            "_chart_steps_done",
            0,
        )
        + 1
    )

    return (
        f"Generated {spec['chart_type']} chart: "
        f"{spec.get('title', 'Analytics Chart')}. "
        f"Artifact: {filename}"
    )


# ============================================================
# Deterministic BQML workflow helpers
# ============================================================

async def _execute_bqml_sql(
    sql: str,
) -> list[dict[str, Any]]:
    """Execute an approved BQML/BigQuery SQL statement and return rows."""

    from .sub_agents.bqml.tools import _get_client

    client = _get_client()

    _logger.info("Executing approved BQML SQL workflow.")

    query_job = client.query(
        sql,
        location="US",
    )

    return [
        dict(row)
        for row in query_job.result()
    ]


async def _save_chart_spec_as_artifact(
    spec: dict[str, Any],
    tool_context: ToolContext,
) -> str:
    """Render a chart spec and persist it as an ADK artifact."""

    is_valid, validation_message = _validate_chart_spec(
        spec
    )

    if not is_valid:
        return (
            "Chart generation failed: "
            f"{validation_message}"
        )

    try:
        png_bytes = _render_chart_to_bytes(
            spec
        )

    except Exception as exc:
        _logger.exception(
            "Deterministic chart rendering failed."
        )

        return (
            "Chart rendering failed: "
            f"{exc}"
        )

    chart_hash = hashlib.sha256(
        _to_json_text(
            spec
        ).encode(
            "utf-8"
        )
    ).hexdigest()[:10]

    filename = (
        f"chart_{int(time.time())}_"
        f"{chart_hash}.png"
    )

    artifact = types.Part.from_bytes(
        data=png_bytes,
        mime_type="image/png",
    )

    try:
        version = await tool_context.save_artifact(
            filename,
            artifact,
        )

    except Exception as exc:
        _logger.exception(
            "Deterministic chart artifact save failed."
        )

        return (
            "Chart was rendered successfully, but artifact saving failed: "
            f"{exc}"
        )

    tool_context.state[
        "last_chart_spec"
    ] = spec

    tool_context.state[
        "last_chart_artifact"
    ] = filename

    tool_context.state[
        "_chart_steps_done"
    ] = (
        tool_context.state.get(
            "_chart_steps_done",
            0,
        )
        + 1
    )

    return (
        f"Generated {spec['chart_type']} chart: "
        f"{spec.get('title', 'Analytics Chart')}. "
        f"Artifact: {filename}"
    )


def _build_revenue_forecast_chart_spec(
    rows: list[dict[str, Any]],
    title: str,
) -> dict[str, Any]:
    """Build a deterministic forecast line-chart specification."""

    return {
        "chart_type": "line",
        "title": title,
        "x_label": "Date",
        "y_label": _forecast_label(),
        "data": {
            "categories": [
                row["date"]
                for row in rows
            ],
            "series": {
                "Forecast": [
                    row["forecast"]
                    for row in rows
                ],
                "Lower Bound": [
                    row["lower"]
                    for row in rows
                ],
                "Upper Bound": [
                    row["upper"]
                    for row in rows
                ],
            },
        },
    }


def _normalize_revenue_forecast_rows(
    raw_rows: list[dict[str, Any]],
    horizon: int,
    explainable: bool = False,
) -> list[dict[str, Any]]:
    """Normalize actual BQML forecast rows and enforce the requested horizon."""

    rows: list[dict[str, Any]] = []

    for item in raw_rows:
        timestamp = (
            item.get("time_series_timestamp")
            if explainable
            else item.get("forecast_timestamp")
        )

        value = (
            item.get("time_series_data")
            if explainable
            else item.get("forecast_value")
        )

        lower = item.get(
            "prediction_interval_lower_bound"
        )
        upper = item.get(
            "prediction_interval_upper_bound"
        )

        if timestamp is None or value is None:
            continue

        if lower is None or upper is None:
            continue

        if explainable and str(
            item.get("time_series_type", "forecast")
        ).lower() != "forecast":
            continue

        row = {
            "forecast_timestamp": str(timestamp),
            "forecast_value": float(value),
            "prediction_interval_lower_bound": float(lower),
            "prediction_interval_upper_bound": float(upper),
        }

        if explainable:
            row.update(
                {
                    "trend": item.get("trend"),
                    "seasonal_period_weekly": item.get(
                        "seasonal_period_weekly"
                    ),
                    "seasonal_period_yearly": item.get(
                        "seasonal_period_yearly"
                    ),
                    "time_series_timestamp": str(timestamp),
                    "time_series_data": float(value),
                    "time_series_type": "forecast",
                }
            )

        rows.append(row)

    rows.sort(
        key=lambda item: item["forecast_timestamp"]
    )

    rows = rows[:horizon]

    if len(rows) != horizon:
        raise RuntimeError(
            "BQML returned "
            f"{len(rows)} forecast rows; expected {horizon}."
        )

    return rows


def _rows_for_forecast_response(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create a compact response representation from forecast rows."""

    compact: list[dict[str, Any]] = []

    for row in rows:
        item = {
            "forecast_timestamp": row["forecast_timestamp"],
            "forecast_value": row["forecast_value"],
            "prediction_interval_lower_bound": row[
                "prediction_interval_lower_bound"
            ],
            "prediction_interval_upper_bound": row[
                "prediction_interval_upper_bound"
            ],
        }

        if "trend" in row:
            item["trend"] = row["trend"]
            item["seasonal_period_weekly"] = row.get(
                "seasonal_period_weekly"
            )
            item["seasonal_period_yearly"] = row.get(
                "seasonal_period_yearly"
            )

        compact.append(item)

    return compact


def _is_revenue_forecast_request(
    request: str,
) -> bool:
    """Return True for forecast requests (the client's one forecasting model)."""

    normalized = re.sub(r"\s+", " ", request.lower()).strip()
    return (
        "forecast" in normalized
        and not re.search(
            r"\b(evaluat\w*|accura\w*|diagnostic\w*|explainable|segment\w*|cluster\w*)\b",
            normalized,
        )
    )


def _is_revenue_evaluation_request(
    request: str,
) -> bool:
    """Return True for forecasting-model evaluation requests."""

    normalized = re.sub(r"\s+", " ", request.lower()).strip()
    return bool(
        re.search(r"\b(evaluat\w*|accura\w*|diagnostic\w*)\b", normalized)
        and re.search(r"\b(revenue|forecast\w*|model|conversions?)\b", normalized)
        and not re.search(r"\b(segment\w*|cluster\w*)\b", normalized)
    )


def _is_model_status_request(
    request: str,
) -> bool:
    """Return True for approved model status requests."""

    normalized = re.sub(
        r"\s+",
        " ",
        request.lower(),
    ).strip()

    return (
        "status" in normalized
        and "model" in normalized
    )


def _is_customer_segmentation_prediction_request(
    request: str,
) -> bool:
    """Return True for customer segmentation prediction requests."""

    normalized = re.sub(
        r"\s+",
        " ",
        request.lower(),
    ).strip()

    return (
        (
            "customer segmentation" in normalized
            or "customer segment" in normalized
            or "customer segments" in normalized
        )
        and (
            "prediction" in normalized
            or "predictions" in normalized
            or "predict" in normalized
        )
        and not _is_customer_segmentation_distribution_request(
            request
        )
    )


def _is_explicit_model_training_request(
    request: str,
) -> bool:
    """Return True for explicit model creation/training requests."""
    from .sub_agents.bqml.tools import training_requested

    return training_requested(request)


async def _run_revenue_forecast(
    request: str,
    tool_context: ToolContext,
    explainable: bool = False,
) -> str:
    """Run an approved revenue forecast deterministically without a second LLM call."""

    from .sub_agents.bqml.tools import (
        get_revenue_explain_forecast_sql,
        get_revenue_forecast_sql,
    )

    horizon = _extract_requested_forecast_horizon(
        request
    ) or 30

    tool_context.state[
        "_bqml_workflow"
    ] = (
        "revenue_explain_forecast"
        if explainable
        else "revenue_forecast"
    )

    sql = (
        get_revenue_explain_forecast_sql(horizon)
        if explainable
        else get_revenue_forecast_sql(horizon)
    )

    raw_rows = await _execute_bqml_sql(
        sql
    )

    normalized_rows = _normalize_revenue_forecast_rows(
        raw_rows,
        horizon,
        explainable=explainable,
    )

    visualization_rows = normalized_rows

    tool_context.state[
        "last_bqml_result"
    ] = _to_json_text(
        _rows_for_forecast_response(
            normalized_rows
        )
    )

    tool_context.state[
        "last_visualization_result"
    ] = visualization_rows

    tool_context.state[
        "last_visualization_source"
    ] = "bqml"

    tool_context.state[
        "last_bqml_sql"
    ] = sql

    compact_rows = _rows_for_forecast_response(
        normalized_rows
    )

    if _is_chart_request(request):
        chart_rows = [
            {
                "date": str(
                    row["forecast_timestamp"]
                )[:10],
                "forecast": float(
                    row["forecast_value"]
                ),
                "lower": float(
                    row["prediction_interval_lower_bound"]
                ),
                "upper": float(
                    row["prediction_interval_upper_bound"]
                ),
            }
            for row in normalized_rows
        ]

        title = (
            f"{horizon}-day Explainable {_forecast_label()} Forecast"
            if explainable
            else f"{horizon}-day {_forecast_label()} Forecast with Prediction Intervals"
        )

        chart_spec = _build_revenue_forecast_chart_spec(
            chart_rows,
            title,
        )

        chart_result = await _save_chart_spec_as_artifact(
            chart_spec,
            tool_context,
        )

        return chart_result

    measure = _forecast_label().lower()
    label = (
        f"explainable {measure} forecast"
        if explainable
        else f"{measure} forecast"
    )

    return (
        f"Here is the {horizon}-day {label}:\n"
        f"{_to_json_text(compact_rows)}"
    )


async def _run_revenue_evaluation(
    tool_context: ToolContext,
) -> str:
    """Run revenue model evaluation deterministically."""

    from .sub_agents.bqml.tools import (
        get_revenue_model_evaluation_sql,
    )

    sql = get_revenue_model_evaluation_sql()

    rows = await _execute_bqml_sql(
        sql
    )

    if not rows:
        raise RuntimeError(
            "Revenue model evaluation returned no rows."
        )

    result_text = _to_json_text(rows)

    tool_context.state[
        "_bqml_workflow"
    ] = "revenue_model_evaluation"
    tool_context.state[
        "last_bqml_result"
    ] = result_text

    return (
        "Revenue model evaluation results:\n"
        f"{result_text}"
    )


async def _run_model_status(
    request: str,
    tool_context: ToolContext,
) -> str:
    """Return factual metadata for an approved model."""

    from .sub_agents.bqml.tools import (
        CUSTOMER_SEGMENT_MODEL_NAME,
        REVENUE_MODEL_NAME,
        get_model_status_sql,
    )

    normalized = request.lower()

    model_name = (
        CUSTOMER_SEGMENT_MODEL_NAME
        if "customer" in normalized
        or "segment" in normalized
        else REVENUE_MODEL_NAME
    )

    result_text = get_model_status_sql(
        model_name
    )

    tool_context.state[
        "_bqml_workflow"
    ] = "model_status"
    tool_context.state[
        "last_bqml_result"
    ] = result_text

    return result_text


async def _run_customer_segmentation_predictions(
    tool_context: ToolContext,
) -> str:
    """Run customer-level segment predictions deterministically."""

    from .sub_agents.bqml.tools import (
        get_customer_segmentation_prediction_sql,
    )

    sql = get_customer_segmentation_prediction_sql()

    rows = await _execute_bqml_sql(
        sql
    )

    if not rows:
        raise RuntimeError(
            "Customer segmentation prediction returned no rows."
        )

    safe_rows = _sanitize_json_value(
        rows
    )

    tool_context.state[
        "_bqml_workflow"
    ] = "customer_segmentation_prediction"
    tool_context.state[
        "last_bqml_result"
    ] = _to_json_text(safe_rows)
    tool_context.state[
        "last_visualization_result"
    ] = safe_rows
    tool_context.state[
        "last_visualization_source"
    ] = "bqml"

    return (
        "Here are the customer segmentation predictions:\n"
        f"{_to_json_text(safe_rows)}"
    )


async def _run_model_training(
    request: str,
    tool_context: ToolContext,
) -> str | None:
    """Run an explicitly requested approved model-training workflow."""

    from .sub_agents.bqml.tools import (
        get_customer_segmentation_training_sql,
        get_revenue_training_sql,
    )

    normalized = request.lower()

    if "customer" in normalized and (
        "segment" in normalized
        or "cluster" in normalized
    ):
        sql = get_customer_segmentation_training_sql()
        workflow = "customer_segmentation_training"
        model_name = "customer_segments"

    elif re.search(r"\b(revenue|forecast\w*|conversions?|model)\b", normalized):
        sql = get_revenue_training_sql()
        workflow = "revenue_model_training"
        model_name = "revenue_forecast"

    else:
        return None

    from .sub_agents.bqml.tools import _get_client, _training_refusal

    # Server-side approval: the USER's message (not the model-written
    # request) must ask for training, and an existing model is only
    # replaced when the user asks to retrain/rebuild it.
    refusal = _training_refusal(
        tool_context.state.get("_current_user_request", ""),
        model_name,
    )
    if refusal:
        return refusal

    client = _get_client()

    _logger.info(
        "Running explicit model training workflow for %s.",
        model_name,
    )

    query_job = client.query(
        sql,
        location="US",
    )
    query_job.result()

    tool_context.state[
        "_bqml_workflow"
    ] = workflow

    result_text = (
        f"Model training completed successfully for "
        f"{model_name}. "
        f"Job ID: {query_job.job_id}"
    )

    tool_context.state[
        "last_bqml_result"
    ] = result_text

    return result_text


# ============================================================
# BQML specialist
# ============================================================

async def call_bqml_agent(
    request: str,
    tool_context: ToolContext,
) -> str:
    """Execute approved BQML workflows with deterministic routing where possible."""

    if not request or not request.strip():
        return "BQML request is empty."

    # ADK State is not a plain dict. Clear the workflow marker by assignment.
    tool_context.state[
        "_bqml_workflow"
    ] = None

    try:
        # 1. Explicit training requests.
        if _is_explicit_model_training_request(request):
            training_result = await _run_model_training(
                request,
                tool_context,
            )

            if training_result is not None:
                return training_result

        # 2. Explainable revenue forecasting.
        if _is_explainable_forecast_request(request):
            return await _run_revenue_forecast(
                request,
                tool_context,
                explainable=True,
            )

        # 3. Customer segment distribution/counts.
        if _is_customer_segmentation_distribution_request(request):
            distribution_result = await _run_customer_segmentation_distribution(
                tool_context
            )

            if _is_chart_request(request):
                aggregate = tool_context.state.get(
                    "last_visualization_result"
                )

                chart_spec = _build_customer_segmentation_distribution_chart_spec(
                    aggregate
                )

                if chart_spec is None:
                    return (
                        f"{distribution_result}\n\n"
                        "Chart generation failed: invalid segment distribution data."
                    )

                chart_result = await _save_chart_spec_as_artifact(
                    chart_spec,
                    tool_context,
                )

                return (
                    f"{chart_result}\n"
                )

            return distribution_result

        # 4. Revenue forecast.
        if _is_revenue_forecast_request(request):
            return await _run_revenue_forecast(
                request,
                tool_context,
                explainable=False,
            )

        # 5. Revenue model evaluation.
        if _is_revenue_evaluation_request(request):
            return await _run_revenue_evaluation(
                tool_context
            )

        # 6. Model status.
        if _is_model_status_request(request):
            return await _run_model_status(
                request,
                tool_context,
            )

        # 7. Customer-level segment prediction.
        if _is_customer_segmentation_prediction_request(request):
            return await _run_customer_segmentation_predictions(
                tool_context
            )

        # ----------------------------------------------------
        # Fallback: retain the BQML specialist for requests
        # outside the approved deterministic routes above.
        # ----------------------------------------------------

        from .sub_agents.bqml.agent import (
            root_agent as bqml_agent,
        )

        agent_tool = AgentTool(
            agent=bqml_agent,
        )

        _logger.info(
            "Calling BQML specialist fallback: %s",
            request[:200],
        )

        result = await agent_tool.run_async(
            args={
                "request": request,
            },
            tool_context=tool_context,
        )

        result_text = str(result)

        tool_context.state[
            "last_bqml_result"
        ] = result_text

        tool_context.state[
            "last_visualization_result"
        ] = result_text

        tool_context.state[
            "last_visualization_source"
        ] = "bqml"

        return result_text

    except Exception as exc:
        _logger.exception(
            "BQML workflow failed."
        )

        failure_message = (
            "BQML request failed: "
            f"{exc}"
        )

        tool_context.state[
            "last_bqml_result"
        ] = failure_message

        return failure_message

