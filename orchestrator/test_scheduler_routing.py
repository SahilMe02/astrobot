import asyncio

from orchestrator.agent import call_scheduler


class FakeContext:
    def __init__(self):
        self.state = {"client_id": "venetian", "_current_client_source": "current_request"}


def test_scheduler_tool_exists():
    from scheduler.agent import root_agent
    assert root_agent.name == "astrobot_scheduler"
