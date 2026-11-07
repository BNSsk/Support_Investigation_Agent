"""Scenario evaluation dataset: 3 NEW cases for each of the 5 preloaded demo scenarios (app/samples.py).

None of these tickets is in the indexed historical base (they were written for evaluation), and none repeats a
demo ticket, so the agent cannot answer by retrieving the ticket itself. Every case except the support-lead ones
sits on a topic with several similar historical cases, so a grounded answer is possible. The support-lead cases
need knowledge the historical base does not contain.

Each case: the steps to run (as in samples.py), the step that is scored (the last one), a reference reply / key
points for that step, and the behaviour the scenario must show (`expect`), checked by the agent_behavior and
tool_usage evaluators.
"""
from __future__ import annotations

SCENARIO_INFO = {
    "normal": ("Normal resolution", "Clear ticket -> grounded resolution from multiple historical cases."),
    "clarify": ("Clarification + resume", "Vague ticket -> agent pauses with a focused question -> resumes with the answer."),
    "memory": ("Long-term memory reuse", "Customer states their environment once; a later, separate conversation reuses it."),
    "growing_kb": ("Retrieval of a newly resolved case",
                   "Resolve + accept a case (indexed as trusted); a similar ticket from another customer retrieves it."),
    "lead_knowledge": ("Support lead teaches a new fix",
                       "The lead's diagnosis (not in the historical base) is applied at review and indexed as "
                       "lead-verified; a similar ticket from another customer gets the lead's fix."),
}

EXPECT = {
    "normal": {"escalation": False, "clarification": False},
    "clarify": {"clarification": True, "tools": ["ask_customer"]},
    "memory": {"memory_reused": True, "clarification": False},
    "growing_kb": {"first_case_indexed": True, "new_case_retrieved": True},
    "lead_knowledge": {"first_case_indexed": True, "lead_case_retrieved": True},
}

CASES = [
    # ------------------------------------------------------------------ 1. normal resolution
    {
        "id": "normal-damaged-delivery", "scenario": "normal",
        "steps": [{
            "customer_id": "cust-harborview-kitchens",
            "subject": "Blender arrived with a cracked jar - need a replacement",
            "body": ("Hello, our order #48213 (ProBlend 900 blender) was delivered yesterday. When we unpacked it the "
                     "glass jar had a long crack and there were glass shards in the box, although the outer carton "
                     "looked fine. We would like a replacement unit as soon as possible rather than a refund. Photos "
                     "are available if needed. Regards, Marco"),
        }],
        "reference": ("Apologise for the damaged delivery; ask for photos of the damage and packaging and confirm the "
                      "order number; arrange a free replacement (ideally expedited) and explain how the damaged unit is "
                      "returned (prepaid return label) or that no return is needed; give the expected timeline and "
                      "offer a refund as the fallback if no replacement is in stock."),
    },
    {
        "id": "normal-vpn-drops", "scenario": "normal",
        "steps": [{
            "customer_id": "cust-meridian-logistics",
            "subject": "VPN drops every 20-30 minutes for our remote staff",
            "body": ("Since last week about 15 of our remote employees lose their VPN connection every 20 to 30 "
                     "minutes and have to reconnect manually. Office staff on the same VPN gateway are not affected. "
                     "Users are on Windows 11 laptops with the latest VPN client. It is disrupting calls and file "
                     "transfers. Can you help us find the cause?"),
        }],
        "reference": ("Check the gateway's idle/session timeout and keep-alive (dead peer detection) settings; review "
                      "client logs of affected users for the disconnect reason; test MTU/fragmentation and NAT/ISP "
                      "timeouts on home networks; try switching the tunnel protocol (UDP/TCP) or split tunnelling; "
                      "collect timestamps and logs and escalate to the network team if the gateway shows tunnel "
                      "renegotiation errors."),
    },
    {
        "id": "normal-excel-macros", "scenario": "normal",
        "steps": [{
            "customer_id": "cust-brightline-accounting",
            "subject": "Excel crashes when opening macro workbooks after Office update",
            "body": ("After the latest Microsoft 365 update rolled out on Monday, Excel crashes immediately whenever "
                     "our finance team opens any of our macro-enabled workbooks (.xlsm). Normal workbooks open fine. "
                     "This affects 8 users and we are blocked on month-end reporting. We already restarted the "
                     "machines."),
        }],
        "reference": ("Treat as urgent (month-end blocked). Start Excel in safe mode and disable add-ins/COM add-ins; "
                      "check Trust Center macro settings and whether the files are blocked as downloaded files; check "
                      "VBA references for missing libraries; repair the Office installation or roll back to the "
                      "previous build / switch update channel temporarily; collect crash event logs and escalate if "
                      "it persists."),
    },
    # ------------------------------------------------------------------ 2. clarification + resume
    {
        "id": "clarify-cant-log-in", "scenario": "clarify",
        "steps": [{
            "customer_id": "cust-pinecrest-dental",
            "subject": "Can't get in",
            "body": "Hi, none of us can get in since yesterday. Please advise what to do.",
            "clarification_answer": ("It's your customer portal. All 6 of our staff accounts fail at the two-factor "
                                     "step: the 6-digit codes from the authenticator app are rejected as invalid. "
                                     "Passwords are accepted. It started yesterday morning, the same day our office "
                                     "PCs and phones switched to daylight saving time."),
        }],
        "reference": ("First ask which system is affected and what error appears. Then: the 2FA codes are time-based, "
                      "so a clock/time-zone offset after the daylight-saving change is the likely cause - set phones "
                      "to automatic date/time, use the authenticator's time sync, check PC/phone time zones; use "
                      "backup codes to get in; an admin can reset the 2FA enrolment; escalate to IT/security with the "
                      "affected accounts if it persists."),
    },
    {
        "id": "clarify-bill-looks-wrong", "scenario": "clarify",
        "steps": [{
            "customer_id": "cust-oakridge-studio",
            "subject": "Question about our bill",
            "body": "Hello, something doesn't look right with what we were charged this time. Can you check it for us?",
            "clarification_answer": ("Our October invoice INV-20931 charges for 25 user licences, but we downgraded "
                                     "to 18 seats on 1 October and received a confirmation email for the downgrade. So "
                                     "we were charged for 7 seats too many."),
        }],
        "reference": ("First ask for the invoice and what looks wrong. Then verify the downgrade date and seat count on "
                      "the account, confirm the overcharge for 7 seats, issue a credit or refund for the difference "
                      "(prorated if applicable) with a corrected invoice, make sure the next cycle bills 18 seats, and "
                      "give a timeline for the refund."),
    },
    {
        "id": "clarify-keeps-disconnecting", "scenario": "clarify",
        "steps": [{
            "customer_id": "cust-lakeside-architects",
            "subject": "Keeps disconnecting",
            "body": "It keeps disconnecting several times a day. Getting really annoying, please fix.",
            "clarification_answer": ("It's our office network printer (HP LaserJet Pro M404). Since we moved it to a "
                                     "new VLAN last week, the Mac users see 'printer offline' several times a day; "
                                     "Windows PCs can still print most of the time. Turning the printer off and on "
                                     "fixes it for a few hours."),
        }],
        "reference": ("First ask which device or service disconnects. Then: give the printer a static IP / DHCP "
                      "reservation; on the Macs add the printer by IP (IPP) instead of Bonjour, because Bonjour/mDNS "
                      "discovery does not cross VLANs (or enable an mDNS reflector); allow printing ports (IPP 631, "
                      "9100) between the VLANs; disable printer sleep/auto-off and update its firmware."),
    },
    # ------------------------------------------------------------------ 3. long-term memory reuse
    {
        "id": "memory-law-firm-outlook", "scenario": "memory",
        "steps": [
            {
                "customer_id": "cust-hartley-legal",
                "subject": "Outlook shared mailboxes show sync errors",
                "body": ("We are a 60-person law firm on your Business plan. All staff use Microsoft 365 with the "
                         "Outlook desktop app on Windows 11, and your document management system files matter emails "
                         "from Outlook automatically through its Outlook add-in. Since Tuesday, Outlook shows 'sync "
                         "error' on our shared mailboxes and filed emails are missing from the matters."),
            },
            {
                "customer_id": "cust-hartley-legal",
                "subject": "Sync errors are back",
                "body": "Hi team, the sync errors came back after this morning's updates. What should we check first?",
                "new_thread": True,
            },
        ],
        "reference": ("Uses the known environment without asking for it again (Microsoft 365, Outlook desktop on "
                      "Windows 11, the document-management Outlook add-in, shared mailboxes): check whether the "
                      "Windows/Office update disabled the add-in and re-enable it; check add-in compatibility with the "
                      "new Office build; rebuild the cached data (OST) for the shared mailboxes; re-authenticate the "
                      "add-in; collect sync logs and escalate with them if it persists."),
    },
    {
        "id": "memory-retailer-inventory", "scenario": "memory",
        "steps": [
            {
                "customer_id": "cust-alpine-outfitters",
                "subject": "Inventory levels not syncing to our online store",
                "body": ("We run our online store on Shopify Plus and our ERP is Microsoft Dynamics 365 Business "
                         "Central, connected through your integration connector. Since we added a second warehouse "
                         "last week, stock levels from the ERP no longer update on the storefront, so we are "
                         "overselling items."),
            },
            {
                "customer_id": "cust-alpine-outfitters",
                "subject": "Stock numbers wrong again",
                "body": "Stock numbers on the storefront are wrong again after the weekend. Where do we start?",
                "new_thread": True,
            },
        ],
        "reference": ("Uses the known environment without asking for it again (Shopify Plus, Dynamics 365 Business "
                      "Central, the integration connector, two warehouses): check the connector's sync job log for "
                      "weekend errors, the warehouse/location mapping (second warehouse included in available stock), "
                      "API token expiry or rate limits; run a manual full inventory sync and verify a few SKUs."),
    },
    {
        "id": "memory-university-zoom-sso", "scenario": "memory",
        "steps": [
            {
                "customer_id": "cust-northgate-university",
                "subject": "Zoom meetings no longer created from tasks",
                "body": ("I'm the IT administrator at a university. About 400 lecturers use your project management "
                         "platform with the Zoom integration, and everyone signs in through SSO with Okta. Since we "
                         "rotated the Okta signing certificate on Friday, scheduling a Zoom meeting from a task fails "
                         "silently and no meeting is created."),
            },
            {
                "customer_id": "cust-northgate-university",
                "subject": "Meetings not being created again",
                "body": "Meetings aren't being created again after this week's maintenance. Any ideas?",
                "new_thread": True,
            },
        ],
        "reference": ("Uses the known environment without asking for it again (project management platform with the "
                      "Zoom integration, Okta SSO, the recent certificate rotation): verify the Okta SAML certificate "
                      "/ metadata in the platform's SSO settings, re-authorise the Zoom integration (OAuth token), "
                      "check the integration log for 401/403 errors and the Zoom app scopes, and test with one "
                      "lecturer account."),
    },
    # ------------------------------------------------------------------ 4. retrieval of a newly resolved case
    {
        "id": "kb-jira-git-links", "scenario": "growing_kb",
        "steps": [
            {
                "customer_id": "cust-vertex-labs",
                "subject": "Jira issues stopped linking to commits after GitLab migration",
                "body": ("We moved all our repositories to a new GitLab group last week. Since then, commits and merge "
                         "requests that mention Jira issue keys (e.g. PAY-142) no longer appear in the Development "
                         "panel of the Jira issues. Before the move it worked fine."),
            },
            {
                "customer_id": "cust-bluewave-software",
                "subject": "Commits not showing up on Jira tickets since repo move",
                "body": ("After we relocated our Git repositories, our commit messages with Jira keys are not linked "
                         "to the tickets any more. How can we restore the link?"),
                "new_thread": True,
            },
        ],
        "reference": ("Reconnect the Git integration in Jira to the new group/location and re-add the moved "
                      "repositories; recreate the webhooks for the new repositories; trigger a re-sync / re-index; "
                      "check the integration account's permissions on the new group; confirm commit messages still "
                      "contain the issue keys."),
    },
    {
        "id": "kb-thermostat-mesh-wifi", "scenario": "growing_kb",
        "steps": [
            {
                "customer_id": "cust-greenleaf-homes",
                "subject": "Smart thermostat hub keeps going offline after new mesh router",
                "body": ("Since we replaced our router with a mesh Wi-Fi system, our smart thermostat hub drops "
                         "offline every evening and the app shows 'device unreachable'. The thermostat itself still "
                         "works locally. Other devices on the Wi-Fi are fine."),
            },
            {
                "customer_id": "cust-summit-property",
                "subject": "Thermostat hub disconnects since installing mesh Wi-Fi",
                "body": ("Our thermostat hub loses its connection a few times a day ever since we installed a mesh "
                         "network. The app says the device is unreachable. What can we do?"),
                "new_thread": True,
            },
        ],
        "reference": ("Smart-home hubs usually need a stable 2.4 GHz connection: create a dedicated 2.4 GHz network or "
                      "disable band steering on the mesh, reserve a fixed IP for the hub, keep it near one mesh node, "
                      "update the hub firmware and re-pair it with the new network."),
    },
    {
        "id": "kb-slack-notification-delay", "scenario": "growing_kb",
        "steps": [
            {
                "customer_id": "cust-copperfield-media",
                "subject": "Task notifications reach Slack 30-60 minutes late",
                "body": ("Since we connected the new Slack app to our project management workspace, notifications "
                         "about task assignments and comments arrive in Slack 30 to 60 minutes late. In-app and email "
                         "notifications are instant. About 50 users are affected."),
            },
            {
                "customer_id": "cust-ridgeway-consulting",
                "subject": "Slack alerts from project boards arrive an hour late",
                "body": ("Alerts from our project boards show up in Slack around an hour after the event. Email alerts "
                         "are on time. Is there a fix?"),
                "new_thread": True,
            },
        ],
        "reference": ("Check the Slack app's notification settings (batching/digest), re-authorise or reinstall the "
                      "Slack integration, verify the webhook/event subscriptions, check for Slack API rate limiting "
                      "and the status page; escalate to the integrations team with timestamps if the delay persists."),
    },
    # ------------------------------------------------------------------ 5. support lead teaches a new fix
    {
        "id": "lead-android-push", "scenario": "lead_knowledge",
        "steps": [
            {
                "customer_id": "cust-fitpulse-app",
                "subject": "Android users stopped receiving push notifications",
                "body": ("Since last Thursday none of our Android users receive push notifications from our app built "
                         "on your mobile SDK. iOS users still get them. Nothing changed in our app release."),
                "lead_root_cause": ("The workspace still sends Android pushes through the legacy FCM server-key API, "
                                    "which Google has shut down; every Android send is now rejected, while iOS goes "
                                    "through APNs and is unaffected."),
                "lead_solution": ("1) In Console > Settings > Push > Android, upload a Firebase service-account JSON "
                                  "to switch to FCM HTTP v1. 2) Update the mobile SDK to 7.3 or later, which registers "
                                  "v1 tokens. 3) Send a test push from Console > Push > Test to confirm delivery."),
            },
            {
                "customer_id": "cust-trailmate",
                "subject": "Push messages not arriving on Android phones",
                "body": ("Our Android app users have not received any push messages for several days, while iPhone "
                         "users still do. We haven't changed anything. Can you help?"),
                "new_thread": True,
            },
        ],
        "reference": ("Root cause: Android pushes still go through the retired legacy FCM server-key API. Fix: upload a "
                      "Firebase service-account JSON in Console > Settings > Push > Android to switch to FCM HTTP v1, "
                      "update the mobile SDK to 7.3+, then send a test push from Console > Push > Test."),
    },
    {
        "id": "lead-emails-to-spam", "scenario": "lead_knowledge",
        "steps": [
            {
                "customer_id": "cust-evergreen-realty",
                "subject": "Our report emails suddenly go to spam",
                "body": ("Since Monday, the scheduled report emails your platform sends from our domain land in "
                         "recipients' spam folders (Gmail and Outlook). Nothing changed on our side except that "
                         "marketing added a new newsletter tool last week."),
                "lead_root_cause": ("Adding the newsletter tool's include pushed the domain's SPF record over the "
                                    "10-DNS-lookup limit, so SPF now fails with PermError and receivers treat the "
                                    "platform's mail as unauthenticated."),
                "lead_solution": ("1) Flatten the SPF record (replace nested includes with the providers' IP ranges) "
                                  "or move the newsletter tool to a subdomain so the root SPF stays under 10 lookups. "
                                  "2) Enable DKIM signing in Settings > Email > Sender domains. 3) Check with an SPF "
                                  "checker that the result is 'pass'."),
            },
            {
                "customer_id": "cust-harbor-insurance",
                "subject": "Platform emails landing in junk since this week",
                "body": ("Emails your platform sends on our behalf started going to junk folders this week. We recently "
                         "started using an extra email marketing service. What is causing this?"),
                "new_thread": True,
            },
        ],
        "reference": ("Root cause: the extra marketing service's SPF include pushed the domain's SPF record over the "
                      "10-DNS-lookup limit (PermError). Fix: flatten the SPF record or move the marketing tool to a "
                      "subdomain, enable DKIM in Settings > Email > Sender domains, and verify SPF passes."),
    },
    {
        "id": "lead-smart-lock-battery", "scenario": "lead_knowledge",
        "steps": [
            {
                "customer_id": "cust-cityloft-rentals",
                "subject": "Smart lock batteries drain in 3 days after firmware 4.2",
                "body": ("After updating our smart locks to firmware 4.2, the batteries are empty after about three "
                         "days instead of several months. We manage 30 apartments with these locks."),
                "lead_root_cause": ("Firmware 4.2 turns on the new Auto-Unlock feature by default, which keeps the "
                                    "Bluetooth beacon scanning continuously and drains the batteries."),
                "lead_solution": ("1) In the app open Lock > Settings > Advanced and switch off Auto-Unlock (or for all "
                                  "locks via Property Manager > Bulk settings). 2) Update to firmware 4.2.1, which fixes "
                                  "the default. 3) Replace the batteries once; normal battery life returns."),
            },
            {
                "customer_id": "cust-maple-suites",
                "subject": "Door lock batteries empty after a few days",
                "body": ("Since the latest firmware update, our door locks need new batteries every few days. Is this "
                         "a known issue?"),
                "new_thread": True,
            },
        ],
        "reference": ("Root cause: firmware 4.2 enables Auto-Unlock by default, keeping the Bluetooth beacon scanning. "
                      "Fix: switch off Auto-Unlock in Lock > Settings > Advanced (or Property Manager > Bulk settings), "
                      "update to firmware 4.2.1, and replace the batteries once."),
    },
]


def examples() -> list[dict]:
    """LangSmith examples: inputs = the steps (the scored step's ticket at top level), outputs = reference."""
    out = []
    for c in CASES:
        last = c["steps"][-1]
        title, desc = SCENARIO_INFO[c["scenario"]]
        out.append({
            "inputs": {"case": c["id"], "scenario": c["scenario"], "scenario_title": title,
                       "subject": last["subject"], "body": last["body"], "steps": c["steps"]},
            "outputs": {"answer": c["reference"], "expect": EXPECT[c["scenario"]]},
            "metadata": {"scenario": c["scenario"], "scenario_description": desc},
        })
    return out
