from orchestrator.agent import _route_with_client


def test_route_with_client_does_not_duplicate_client_marker():
    request = "[Client: venetian] [Client: venetian] Check recent emails for blockers."
    routed = _route_with_client(request, "venetian")
    assert routed.count("[Client: venetian]") == 1
    assert routed.endswith("Check recent emails for blockers.")
