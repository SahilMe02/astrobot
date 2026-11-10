"""Create SYNTHETIC per-client performance tables for development.

For each client in tenancy/clients.json this creates, in GOOGLE_CLOUD_PROJECT
(location US):
  astrobot_<client>.performance   daily Date x Channel x Campaign metrics
  astrobot_ml_<client>            empty dataset for that client's BQML models

The numbers are generated, not real client data. Re-running replaces the
tables with the same deterministic data.

Usage:
  uv run python scripts/create_synthetic_clients.py
"""

from __future__ import annotations

import math
import os
import random
import sys
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import bigquery

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")  # same project settings the app uses
sys.path.insert(0, str(ROOT))
from tenancy import data_sources, registry  # noqa: E402

DAYS = 730
LOCATION = "US"

# Per-client channel mix: channel -> (daily spend, CPC, conversion rate, CTR)
PROFILES = {
    "npi": {
        "value_per_conversion": 160.0,
        "channels": {
            "Paid Search": (1800, 1.9, 0.045, 0.040),
            "Paid Social": (1400, 0.9, 0.012, 0.011),
            "Display": (900, 0.6, 0.004, 0.003),
            "Online Video": (1100, 1.2, 0.006, 0.006),
            "Email": (150, 0.1, 0.030, 0.020),
            "Metasearch": (700, 1.4, 0.050, 0.030),
        },
    },
    "venetian": {
        "value_per_conversion": 180.0,
        "channels": {
            "Branded Search": (2200, 1.2, 0.080, 0.090),
            "Non-Brand Search": (1600, 2.8, 0.025, 0.030),
            "Display": (1200, 0.7, 0.004, 0.003),
            "Paid Social": (1500, 1.0, 0.010, 0.012),
            "CTV": (1300, 3.5, 0.008, 0.004),
            "OOH": (800, 4.0, 0.003, 0.002),
        },
    },
    "winndixie": {
        "value_per_conversion": 38.0,
        "channels": {
            "Paid Search": (1300, 0.9, 0.060, 0.050),
            "Paid Social": (1700, 0.6, 0.020, 0.015),
            "OTT": (1500, 2.5, 0.010, 0.005),
            "Display": (900, 0.5, 0.008, 0.004),
            "Streaming Audio": (600, 1.8, 0.006, 0.003),
            "Retail Media": (1100, 0.8, 0.070, 0.035),
        },
    },
}

SCHEMA = [
    bigquery.SchemaField("Date", "DATE"),
    bigquery.SchemaField("Client", "STRING"),
    bigquery.SchemaField("Channel", "STRING"),
    bigquery.SchemaField("Campaign", "STRING"),
    bigquery.SchemaField("Cost", "FLOAT64"),
    bigquery.SchemaField("Impressions", "INT64"),
    bigquery.SchemaField("Clicks", "INT64"),
    bigquery.SchemaField("Conversions", "INT64"),
    bigquery.SchemaField("Revenue", "FLOAT64"),
]


def generate(client_id: str, end: date) -> list[dict]:
    profile = PROFILES[client_id]
    rng = random.Random(f"astrobot-{client_id}")
    rows = []
    start = end - timedelta(days=DAYS - 1)
    for offset in range(DAYS):
        day = start + timedelta(days=offset)
        season = 1 + 0.25 * math.sin(2 * math.pi * (day.timetuple().tm_yday - 80) / 365)
        weekday = 1.12 if day.weekday() >= 4 else 0.95
        for channel, (spend, cpc, cvr, ctr) in profile["channels"].items():
            for campaign in ("Always On", "Promo"):
                share = 0.7 if campaign == "Always On" else 0.3
                cost = spend * share * season * weekday * rng.uniform(0.8, 1.2)
                clicks = max(0, int(cost / (cpc * rng.uniform(0.85, 1.15))))
                impressions = max(clicks, int(clicks / (ctr * rng.uniform(0.8, 1.2))))
                conversions = max(0, int(clicks * cvr * season * rng.uniform(0.7, 1.3)))
                revenue = conversions * profile["value_per_conversion"] * rng.uniform(0.85, 1.15)
                rows.append({
                    "Date": day.isoformat(),
                    "Client": client_id,
                    "Channel": channel,
                    "Campaign": f"{channel} - {campaign}",
                    "Cost": round(cost, 2),
                    "Impressions": impressions,
                    "Clicks": clicks,
                    "Conversions": conversions,
                    "Revenue": round(revenue, 2),
                })
    return rows


def main() -> None:
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:
        sys.exit("GOOGLE_CLOUD_PROJECT is not set in .env or the shell.")
    bq = bigquery.Client(project=project)
    print(f"Creating synthetic client tables in project: {project}")
    end = date.today() - timedelta(days=1)

    for client_id in registry.clients():
        if client_id not in PROFILES:
            print(f"skip {client_id}: no synthetic profile")
            continue
        source = data_sources.for_client(client_id)
        table_id = source.tables["performance"]
        data_dataset = table_id.rsplit(".", 1)[0]

        for dataset_id in (data_dataset, source.bqml_dataset):
            ds = bigquery.Dataset(dataset_id)
            ds.location = LOCATION
            ds.description = f"AstroBot SYNTHETIC development data for {client_id}."
            bq.create_dataset(ds, exists_ok=True)

        rows = generate(client_id, end)
        job = bq.load_table_from_json(
            rows,
            table_id,
            job_config=bigquery.LoadJobConfig(
                schema=SCHEMA,
                write_disposition="WRITE_TRUNCATE",
            ),
        )
        job.result()
        table = bq.get_table(table_id)
        table.description = "SYNTHETIC data generated by scripts/create_synthetic_clients.py. Not real client data."
        bq.update_table(table, ["description"])
        print(f"{client_id}: {table.num_rows} rows -> {table_id}")


if __name__ == "__main__":
    main()
