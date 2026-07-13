"""Customer-support benchmark scenarios.

Each scenario is a measurable task: a goal, the tools available, the tools a
correct solution should call (for precision/recall), and declarative checks for
task success. All tool responses are deterministic (see tools/support_tools.py),
so a model either drives them correctly or it doesn't.
"""

from benchmarks.core.checks import (
    called,
    content_contains,
    content_matches,
    no_tools,
    not_called,
)
from benchmarks.core.scenario import Scenario, Turn

_ALL_TOOLS = [
    "get_user_profile", "get_transaction_history", "check_fraud_score",
    "block_account", "process_refund", "create_ticket",
]


SCENARIOS = [
    Scenario(
        id="profile_lookup",
        description="Answer an account question by looking up the profile (single tool).",
        goal="What tier is the account platinum.customer@bank.com and what's the balance?",
        tools=_ALL_TOOLS,
        expected_tools=["get_user_profile"],
        checks=[
            called("get_user_profile"),
            content_contains("platinum"),
            content_matches(r"50[,.\s]?000"),   # 50000 / 50,000 / 50 000
        ],
    ),

    Scenario(
        id="fraud_block",
        description="Investigate a fraud report and block the account (multi-tool sequence).",
        goal=("I'm fraud.victim@bank.com and I see a charge I didn't make. "
              "Please check my account and block it if it's fraud."),
        tools=_ALL_TOOLS,
        expected_tools=["check_fraud_score", "block_account"],
        checks=[
            called("check_fraud_score"),
            called("block_account"),
            content_contains("block"),
        ],
    ),

    Scenario(
        id="refund_flow",
        description="Find the suspicious transaction, then refund it (dependent tool calls).",
        goal=("This is fraud.victim@bank.com. Look at my recent transactions and "
              "refund the suspicious one."),
        tools=_ALL_TOOLS,
        expected_tools=["get_transaction_history", "process_refund"],
        checks=[
            called("get_transaction_history"),
            called("process_refund"),
            content_contains("refund"),
        ],
    ),

    Scenario(
        id="smalltalk_no_tools",
        description="A pure greeting must be answered WITHOUT calling any tool (over-acting guard).",
        goal="Hi there! Just wanted to say thanks for the great service yesterday.",
        tools=_ALL_TOOLS,
        expected_tools=[],
        checks=[
            no_tools(),
        ],
    ),

    Scenario(
        id="memory_recall",
        description="Use prior conversation context: recall the customer's name from history.",
        goal="Can you remind me what tier I'm on again?",
        tools=_ALL_TOOLS,
        expected_tools=["get_user_profile"],
        history=[
            Turn("user", "Hi, I'm Carlos and my email is standard.user@bank.com."),
            Turn("assistant", "Hello Carlos, how can I help you today?"),
        ],
        checks=[
            content_contains("Carlos"),
            called("get_user_profile"),
        ],
    ),

    Scenario(
        id="banned_account_guard",
        description="A banned account must not be operated on; open a ticket instead of blocking.",
        goal=("I'm banned.user@bank.com and I want to make a big transfer right now.")
             ,
        tools=_ALL_TOOLS,
        expected_tools=["get_user_profile"],
        checks=[
            called("get_user_profile"),
            not_called("process_refund"),
            content_contains("banned"),
        ],
    ),
]
