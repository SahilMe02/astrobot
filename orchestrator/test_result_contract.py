from orchestrator.result_contract import (
    make_error_result,
    make_success_result,
    make_unavailable_result,
    store_specialist_result,
)


def test_success_result_contains_common_fields():
    result = make_success_result(
        "economist",
        "Flight prices are elevated.",
        client_id="venetian",
        metadata={"client_source": "current_request"},
    )

    payload = result.to_dict()

    assert payload["specialist"] == "economist"
    assert payload["status"] == "success"
    assert payload["output"] == "Flight prices are elevated."
    assert payload["error"] == ""
    assert payload["client_id"] == "venetian"
    assert payload["metadata"]["client_source"] == "current_request"


def test_error_result_contains_failure_without_fake_output():
    result = make_error_result(
        "data_science",
        "BigQuery request failed.",
    )

    payload = result.to_dict()

    assert payload["specialist"] == "data_science"
    assert payload["status"] == "error"
    assert payload["output"] == ""
    assert payload["error"] == "BigQuery request failed."


def test_store_specialist_result_keeps_previous_specialists():
    state = {}

    store_specialist_result(
        state,
        make_success_result("data_science", "Revenue is 100."),
    )
    store_specialist_result(
        state,
        make_success_result("economist", "Fuel costs are elevated."),
    )

    assert set(state["specialist_results"]) == {
        "data_science",
        "economist",
    }
    assert state["last_specialist"] == "economist"
    assert state["last_specialist_result"]["specialist"] == "economist"


def test_result_can_be_serialized_for_internal_use():
    result = make_success_result(
        "persona_aggregator",
        "High rollers prefer premium positioning.",
    )

    payload = result.to_json()

    assert '"specialist": "persona_aggregator"' in payload
    assert '"status": "success"' in payload


def test_unavailable_result_is_distinct_from_error():
    result = make_unavailable_result(
        "project_manager",
        "Gmail is not configured.",
        client_id="venetian",
        reason="missing_required_configuration",
    )

    payload = result.to_dict()

    assert payload["status"] == "unavailable"
    assert payload["output"] == "Gmail is not configured."
    assert payload["error"] == "missing_required_configuration"
    assert payload["client_id"] == "venetian"
