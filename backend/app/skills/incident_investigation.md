---
name: incident_investigation
description: Technical troubleshooting for incidents/problems - errors, outages, bugs, performance, connectivity, security events.
stage: investigate
applies_to: Incident, Problem, Technical Support, IT Support, Service Outages and Maintenance, Product Support
---
Approach this like a tier-2 engineer working only from past case records.

1. State a working hypothesis (likely root cause) after reading the initial evidence.
2. Test it: run at least one targeted search using the specific symptom / error / product terms,
   optionally filtered by the predicted queue or a distinctive tag.
3. Open the full record (get_ticket_by_id) of the 1-2 most promising cases to read their complete
   resolutions before relying on them.
4. Compare resolutions across cases:
   - consensus steps (appear in 2+ cases) -> strong evidence
   - single-case steps -> weak evidence, mention as optional
   - contradictory steps -> record the conflict and what would disambiguate it
5. Check customer history for repeats: a recurring issue is a signal for Problem type or escalation.
6. Security incidents, data loss, or multi-user outages require escalation even if a fix is known.
7. Never invent product-specific settings, version numbers or URLs that do not appear in evidence.
