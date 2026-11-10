from project_manager.agent import root_agent
from project_manager.tools.gmail_tool import _authorized_client, _configuration_error


def test_project_manager_imports_and_has_expected_name():
    assert root_agent.name == "project_manager_agent"


def test_client_boundary_allows_matching_session_client():
    from tenancy.context import client_scope

    with client_scope("venetian"):
        ok, reason = _authorized_client("venetian", None)
    assert ok is True
    assert reason == ""


def test_client_boundary_blocks_cross_client_access():
    from tenancy.context import client_scope

    with client_scope("venetian"):
        ok, reason = _authorized_client("npi", None)
    assert ok is False
    assert "venetian" in reason


def test_client_boundary_ignores_state_without_server_scope():
    class Context:
        state = {"client_id": "venetian"}

    ok, _ = _authorized_client("venetian", Context())
    assert ok is False


def test_missing_gmail_configuration_is_explicit():
    # The local test environment intentionally does not require live Gmail credentials.
    message = _configuration_error()
    assert message is None or "Project Manager Gmail is not configured yet" in message
