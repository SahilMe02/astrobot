# AstroBot Scheduler

Current phase: local schedule control plane.

The scheduler stores client-scoped schedule definitions in `schedules.json` and exposes create/list/update/delete operations through an ADK specialist.

Production target: replace the local store/execution layer with the reference Firestore + Cloud Scheduler + Cloud Tasks architecture without changing the agent/tool contract.
