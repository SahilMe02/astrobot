from orchestrator.result_contract import make_unavailable_result


def test_unavailable_result_has_explicit_status():
    result = make_unavailable_result(
        "project_manager",
        "Project Manager Gmail is not configured yet.",
        client_id="venetian",
        metadata={"reason": "missing_required_configuration"},
    )

    payload = result.to_dict()

    assert payload["specialist"] == "project_manager"
    assert payload["status"] == "unavailable"
    assert payload["output"] == "Project Manager Gmail is not configured yet."
    assert payload["error"] == "missing_required_configuration"
    assert payload["client_id"] == "venetian"
