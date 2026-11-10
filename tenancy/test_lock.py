import json

import pytest

from tenancy import lock, registry


@pytest.fixture(autouse=True)
def dev_files(tmp_path, monkeypatch):
    clients = {"clients": {
        "npi": {"display_name": "NPI", "aliases": ["npi"]},
        "venetian": {"display_name": "The Venetian", "aliases": ["venetian", "the venetian"]},
        "winndixie": {"display_name": "Winn-Dixie", "aliases": ["winn-dixie", "winn dixie"]},
    }}
    access = {"users": {"analyst": ["npi", "venetian"], "nobody": []}}
    (tmp_path / "c.json").write_text(json.dumps(clients))
    (tmp_path / "a.json").write_text(json.dumps(access))
    monkeypatch.setenv("ASTROBOT_CLIENTS_FILE", str(tmp_path / "c.json"))
    monkeypatch.setenv("ASTROBOT_ACCESS_FILE", str(tmp_path / "a.json"))
    registry.reload()
    yield
    registry.reload()


def test_first_authorised_client_locks():
    d = lock.decide("[Client: npi] revenue last month", "", "analyst")
    assert (d.client_id, d.newly_locked, d.blocked) == ("npi", True, False)


def test_alias_is_recognised():
    assert lock.decide("How is The Venetian doing?", "", "analyst").client_id == "venetian"


def test_switch_after_lock_is_blocked():
    d = lock.decide("Show me Venetian revenue", "npi", "analyst")
    assert d.blocked and "locked to NPI" in d.block_message


def test_explicit_marker_switch_is_blocked():
    assert lock.decide("[Client: venetian] revenue", "npi", "analyst").blocked


def test_generic_follow_up_keeps_lock():
    d = lock.decide("And what about last week?", "npi", "analyst")
    assert (d.client_id, d.newly_locked, d.blocked) == ("npi", False, False)


def test_same_client_repeated_is_allowed():
    assert lock.decide("[Client: npi] again", "npi", "analyst").client_id == "npi"


def test_unknown_explicit_client_is_blocked():
    assert lock.decide("[Client: acme] revenue", "", "analyst").blocked


def test_unknown_explicit_client_blocked_even_when_locked():
    assert lock.decide("[Client: acme] revenue", "npi", "analyst").blocked


def test_two_clients_in_first_message_is_blocked():
    assert lock.decide("Compare NPI and Venetian", "", "analyst").blocked


def test_unauthorised_client_is_blocked():
    d = lock.decide("Winn-Dixie revenue", "", "analyst")
    assert d.blocked and "do not have access" in d.block_message


def test_unknown_user_has_no_access():
    assert lock.decide("[Client: npi] revenue", "", "stranger").blocked


def test_empty_access_list_means_no_access():
    assert lock.decide("[Client: npi] revenue", "", "nobody").blocked


def test_forged_lock_to_unauthorised_client_is_blocked():
    # State says winndixie, but this user is not allowed winndixie.
    assert lock.decide("revenue please", "winndixie", "analyst").blocked


def test_forged_lock_to_unknown_client_is_blocked():
    assert lock.decide("revenue please", "acme", "analyst").blocked


def test_no_client_is_allowed_for_generic_questions():
    d = lock.decide("What are current flight price trends?", "", "analyst")
    assert (d.client_id, d.blocked) == ("", False)


def test_missing_registry_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("ASTROBOT_CLIENTS_FILE", str(tmp_path / "missing.json"))
    registry.reload()
    assert registry.clients() == {}
    assert lock.decide("[Client: npi] revenue", "", "analyst").blocked
