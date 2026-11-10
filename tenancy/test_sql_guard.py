import pytest

from tenancy.sql_guard import check

NPI = "proj-1.astrobot_npi.performance"
ALLOWED = {NPI}


@pytest.mark.parametrize("sql", [
    f"SELECT Channel, SUM(Cost) FROM `{NPI}` GROUP BY 1",
    f"SELECT SUM(Cost) FROM `proj-1`.astrobot_npi.performance",
    f"WITH t AS (SELECT Channel, Cost FROM `{NPI}`) SELECT Channel, SUM(Cost) FROM t GROUP BY 1",
    f"SELECT 1 AS x FROM `{NPI}` UNION ALL SELECT 2 FROM `{NPI}`",
    f"SELECT 1 FROM `{NPI}` -- astrobot_venetian in a comment is harmless",
])
def test_allowed(sql):
    assert check(sql, ALLOWED) == (True, "")


@pytest.mark.parametrize("sql", [
    "SELECT * FROM `proj-1.astrobot_venetian.performance`",
    f"SELECT * FROM `{NPI}` WHERE Channel IN (SELECT Channel FROM `proj-1.astrobot_venetian.performance`)",
    f"SELECT * FROM `{NPI}` a, `proj-1.astrobot_venetian.performance` b",
    "WITH performance AS (SELECT 1) SELECT * FROM `proj-1.astrobot_venetian.performance`",
    "SELECT * FROM performance",
    "SELECT * FROM astrobot_npi.performance",
    "SELECT * FROM `proj-1.astrobot_*.performance`",
    "SELECT * FROM `proj-1.astrobot_npi.INFORMATION_SCHEMA.TABLES`",
    f"SELECT 1 FROM `{NPI}`; SELECT 2 FROM `{NPI}`",
    f"DELETE FROM `{NPI}` WHERE TRUE",
    "CREATE TABLE `proj-1.astrobot_npi.x` AS SELECT 1",
    "CREATE OR REPLACE MODEL `proj-1.astrobot_ml_npi.m` AS SELECT 1",
    f"INSERT INTO `{NPI}` (Channel) VALUES ('x')",
    "SELECT * FROM EXTERNAL_QUERY('conn', 'select 1')",
    "SELECT * FROM ML.FORECAST(MODEL `proj-1.astrobot_ml_venetian.revenue_forecast`, STRUCT(30 AS horizon))",
    "",
    "not sql at all (",
])
def test_blocked(sql):
    ok, reason = check(sql, ALLOWED)
    assert not ok and reason
