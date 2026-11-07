"""Demo scenarios required by the assessment (also used by scripts/demo.py and the UI)."""

SCENARIOS = [
    {
        "id": "normal",
        "title": "1. Normal resolution",
        "description": "Clear billing ticket -> grounded resolution from multiple historical cases.",
        "steps": [{
            "customer_id": "cust-northwind",
            "subject": "Charged twice for this month's subscription",
            "body": ("Hello, I noticed two identical charges for our monthly subscription on my credit card "
                     "statement this month. We only have one account. Could you please check and refund the "
                     "duplicate payment? Thanks, Anna"),
        }],
    },
    {
        "id": "clarify",
        "title": "2. Clarification + resume",
        "description": "Vague ticket -> agent pauses with a focused question -> resumes with the answer.",
        "steps": [{
            "customer_id": "cust-contoso",
            "subject": "Not working",
            "body": "Hi, it stopped working again this morning. Can you help me fix it?",
            "clarification_answer": ("It's the integration between QuickBooks Online and our data analytics "
                                     "dashboard. Since yesterday's password reset the sync fails with an "
                                     "authentication error. Only my account is affected."),
        }],
    },
    {
        "id": "memory",
        "title": "3. Long-term memory reuse",
        "description": "Customer states their environment once; a later, separate conversation reuses it.",
        "steps": [
            {
                "customer_id": "cust-stmarys-hospital",
                "subject": "EMR integration sync delays with telemedicine platform",
                "body": ("We are a hospital on your Enterprise plan. Our EMR integration module runs on Azure-hosted "
                         "servers and syncs patient appointments to the telemedicine platform. Since Monday the "
                         "sync is delayed by 30-40 minutes. No error messages, the servers look healthy."),
            },
            {
                "customer_id": "cust-stmarys-hospital",
                "subject": "Sync delays are back",
                "body": "Hi team, the sync delays started again after last night's maintenance window. What should we check?",
                "new_thread": True,
            },
        ],
    },
    {
        "id": "growing_kb",
        "title": "4. Retrieval of a newly resolved case",
        "description": "Resolve + accept a case (indexed as trusted), then a similar ticket from another customer retrieves it.",
        "steps": [
            {
                "customer_id": "cust-globex",
                "subject": "Charged again after cancelling our subscription",
                "body": ("We cancelled our team subscription at the end of last month and received the cancellation "
                         "email, but our card was still charged for this month. Please refund this charge."),
            },
            {
                "customer_id": "cust-initech",
                "subject": "Billed after cancellation - need a refund",
                "body": ("Our subscription was cancelled a few weeks ago (we have the confirmation email), yet we were "
                         "billed again this month. How do we get this charge refunded?"),
                "new_thread": True,
            },
        ],
    },
    {
        "id": "lead_knowledge",
        "title": "5. Support lead teaches a new fix",
        "description": ("The lead's diagnosis (not in the historical base) is applied at review and indexed as "
                        "lead-verified; a similar ticket from another customer gets the lead's fix."),
        "steps": [
            {
                "customer_id": "cust-fabrikam",
                "subject": "Premium analytics dashboard stops refreshing every afternoon",
                "body": ("Since we moved to the Premium plan, our analytics dashboard stops refreshing QuickBooks data "
                         "every afternoon between 2 and 5 pm and shows 'request failed' on most widgets. Mornings "
                         "are fine. About 40 people use the dashboard. What is going on?"),
                # demo values for the review card (the lead's own knowledge; no historical case contains it)
                "lead_root_cause": ("Premium dashboards refresh every widget for every viewer once a minute. With ~40 "
                                    "concurrent users in the afternoon peak, the QuickBooks connector exceeds its "
                                    "requests-per-second (RPS) limit and the integration throttles the requests."),
                "lead_solution": ("1) In Dashboard Settings > Data Sources > QuickBooks, switch the refresh mode to "
                                  "'Shared cache' and set the interval to 15 minutes. 2) If the account needs live "
                                  "data, the account admin requests an RPS quota increase from the integrations team."),
            },
            {
                "customer_id": "cust-tailspin",
                "subject": "Dashboard widgets fail during busy hours",
                "body": ("Our Premium dashboard with the QuickBooks data source works in the morning, but from early "
                         "afternoon most widgets show 'request failed' until the evening. We have around 35 users."),
                "new_thread": True,
            },
        ],
    },
]
