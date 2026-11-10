from project_manager.agent import PROJECT_MANAGER_INSTRUCTION, _is_email_analysis_request


def test_project_manager_uses_client_scoped_discovery_for_semantic_requests():
    assert 'fetch_recent_threads for the active client with an EMPTY query' in PROJECT_MANAGER_INSTRUCTION
    assert 'extract_thread_snippets' in PROJECT_MANAGER_INSTRUCTION
    assert 'get_thread_messages_full' in PROJECT_MANAGER_INSTRUCTION


def test_project_manager_does_not_rely_on_literal_blocker_phrase():
    assert 'Do not require literal words like "blocker" or "pending approval"' in PROJECT_MANAGER_INSTRUCTION


def test_email_analysis_request_detection():
    assert _is_email_analysis_request(
        "Check recent client emails for blockers and pending approvals."
    ) is True
    assert _is_email_analysis_request(
        "Give me the project revenue forecast."
    ) is False
