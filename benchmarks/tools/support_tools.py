"""Deterministic mock tools for the benchmark.

Every tool returns a fixed response derived only from its inputs — no network,
no randomness, no clock. This makes a run's outcome a function of the *model's
decisions alone*, so differences between models/agents are attributable to the
model, not to flaky tool behavior.

User state is encoded in the email keyword:
    *plat*   -> PLATINUM tier, high balance, clean history
    *fraud*  -> flagged account, a suspicious recent transaction
    *banned* -> KYC BANNED, no operations allowed
    (other)  -> STANDARD tier
"""

import json
from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


def _profile_for(email: str) -> dict:
    e = email.lower()
    if "plat" in e:
        return {"email": email, "kyc": "VERIFIED", "tier": "PLATINUM",
                "balance": 50000, "credit_score": 850}
    if "fraud" in e:
        return {"email": email, "kyc": "VERIFIED", "tier": "GOLD",
                "balance": 3200, "credit_score": 690}
    if "banned" in e:
        return {"email": email, "kyc": "BANNED", "tier": "NONE",
                "balance": 0, "credit_score": 0}
    return {"email": email, "kyc": "VERIFIED", "tier": "STANDARD",
            "balance": 1200, "credit_score": 720}


# ---------------------------------------------------------------------------
# get_user_profile
# ---------------------------------------------------------------------------

class _ProfileArgs(BaseModel):
    email: str = Field(..., description="Customer email address")


class GetUserProfile(BaseTool):
    name: str = "get_user_profile"
    description: str = "Look up a customer profile: KYC status, tier, balance and credit score."
    args_schema: Type[BaseModel] = _ProfileArgs

    def _run(self, email: str) -> str:
        return json.dumps(_profile_for(email))


# ---------------------------------------------------------------------------
# get_transaction_history
# ---------------------------------------------------------------------------

class _HistoryArgs(BaseModel):
    email: str = Field(..., description="Customer email address")


class GetTransactionHistory(BaseTool):
    name: str = "get_transaction_history"
    description: str = "Return the recent transaction history for a customer."
    args_schema: Type[BaseModel] = _HistoryArgs

    def _run(self, email: str) -> str:
        if "fraud" in email.lower():
            txns = [
                {"id": "TX-100", "amount": 42.0, "merchant": "Coffee Bar", "status": "ok"},
                {"id": "TX-777", "amount": 4200.0, "merchant": "Unknown-Overseas", "status": "suspicious"},
            ]
        else:
            txns = [
                {"id": "TX-100", "amount": 42.0, "merchant": "Coffee Bar", "status": "ok"},
                {"id": "TX-101", "amount": 18.5, "merchant": "Bookstore", "status": "ok"},
            ]
        return json.dumps(txns)


# ---------------------------------------------------------------------------
# check_fraud_score
# ---------------------------------------------------------------------------

class _FraudArgs(BaseModel):
    email: str = Field(..., description="Customer email address")


class CheckFraudScore(BaseTool):
    name: str = "check_fraud_score"
    description: str = "Compute a 0-100 fraud risk score for the customer's account."
    args_schema: Type[BaseModel] = _FraudArgs

    def _run(self, email: str) -> str:
        score = 92 if "fraud" in email.lower() else 8
        return json.dumps({"email": email, "fraud_score": score,
                           "risk": "HIGH" if score >= 70 else "LOW"})


# ---------------------------------------------------------------------------
# block_account
# ---------------------------------------------------------------------------

class _BlockArgs(BaseModel):
    email: str = Field(..., description="Customer email address")
    reason: str = Field(..., description="Reason for blocking the account")


class BlockAccount(BaseTool):
    name: str = "block_account"
    description: str = "Immediately block a customer account. Use only for confirmed fraud or a customer request."
    args_schema: Type[BaseModel] = _BlockArgs

    def _run(self, email: str, reason: str) -> str:
        return json.dumps({"email": email, "status": "BLOCKED", "reason": reason})


# ---------------------------------------------------------------------------
# process_refund
# ---------------------------------------------------------------------------

class _RefundArgs(BaseModel):
    email: str = Field(..., description="Customer email address")
    transaction_id: str = Field(..., description="ID of the transaction to refund")
    amount: float = Field(..., description="Amount to refund")


class ProcessRefund(BaseTool):
    name: str = "process_refund"
    description: str = "Refund a specific transaction back to the customer."
    args_schema: Type[BaseModel] = _RefundArgs

    def _run(self, email: str, transaction_id: str, amount: float) -> str:
        return json.dumps({"email": email, "transaction_id": transaction_id,
                           "refunded": amount, "status": "REFUNDED"})


# ---------------------------------------------------------------------------
# create_ticket
# ---------------------------------------------------------------------------

class _TicketArgs(BaseModel):
    email: str = Field(..., description="Customer email address")
    subject: str = Field(..., description="Short subject describing the issue")


class CreateTicket(BaseTool):
    name: str = "create_ticket"
    description: str = "Open a support ticket for an issue that needs human follow-up."
    args_schema: Type[BaseModel] = _TicketArgs

    def _run(self, email: str, subject: str) -> str:
        return json.dumps({"ticket_id": "TCK-4242", "email": email,
                           "subject": subject, "status": "OPEN"})


# Registry: name -> tool factory
_ALL = [
    GetUserProfile, GetTransactionHistory, CheckFraudScore,
    BlockAccount, ProcessRefund, CreateTicket,
]
_BY_NAME = {cls().name: cls for cls in _ALL}

ALL_TOOL_NAMES = sorted(_BY_NAME.keys())


def build_tools(names):
    """Instantiate the named tools. Unknown names raise KeyError."""
    return [_BY_NAME[name]() for name in names]
