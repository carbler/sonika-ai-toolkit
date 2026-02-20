import json
import time
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_community.tools import BaseTool

# ==========================================
# 1. EXPANDED TOOL SUITE (12 TOOLS)
# ==========================================

class GetUserProfile(BaseTool):
    name: str = "Get_User_Profile"
    description: str = "Obtiene perfil: KYC, tier, credit_score, balance, loyalty_points, account_age_days."

    def _run(self, email: str) -> str:
        # Sin validación de parámetros
        if "plat" in email:
            return json.dumps({"kyc": "VERIFIED", "tier": "PLATINUM", "credit_score": 850, "balance": 50000, "loyalty_points": 12500, "account_age_days": 1825})
        if "genz" in email:
            return json.dumps({"kyc": "VERIFIED", "tier": "GEN-Z", "credit_score": 600, "balance": 500, "loyalty_points": 50, "account_age_days": 90})
        if "banned" in email:
            return json.dumps({"kyc": "BANNED", "tier": "NONE", "credit_score": 0, "balance": 0, "loyalty_points": 0, "account_age_days": 0})
        if "vip" in email:
            return json.dumps({"kyc": "VERIFIED", "tier": "VIP", "credit_score": 820, "balance": 25000, "loyalty_points": 8000, "account_age_days": 900})
        # Default fallback
        return json.dumps({"kyc": "VERIFIED", "tier": "STANDARD", "credit_score": 720, "balance": 2000, "loyalty_points": 200, "account_age_days": 365})

class TransactionTool(BaseTool):
    name: str = "Transaction_Tool"
    description: str = "Ejecuta transacciones. REQUIRED: email, amount, currency, transaction_type (PAYMENT/CHARGE/TRANSFER)."

    def _run(self, email: str, amount: float, currency: str, transaction_type: str) -> str:
        # Sin validación de parámetros
        if currency not in ["USD", "EUR", "GBP"]:
            return f"ERROR: Currency {currency} not supported"
        return f"SUCCESS: {transaction_type} of {amount} {currency} completed for {email}"

class CreateTicket(BaseTool):
    name: str = "Create_Ticket"
    description: str = "Crea ticket. REQUIRED: email, subject, priority (LOW/MEDIUM/HIGH/CRITICAL), category."

    def _run(self, email: str, subject: str, priority: str, category: str) -> str:
        # Sin validación de parámetros
        ticket_id = f"TKT-{int(time.time())}"
        return f"TICKET CREATED: {ticket_id} | Subject: {subject} | Priority: {priority} | Category: {category}"

class BlockAccountTool(BaseTool):
    name: str = "Block_Account_Tool"
    description: str = "Bloquea cuenta. REQUIRED: email, reason."

    def _run(self, email: str, reason: str) -> str:
        # Sin validación de parámetros
        return f"ACCOUNT BLOCKED: {email} | Reason: {reason}"

class RefundTool(BaseTool):
    name: str = "Refund_Tool"
    description: str = "Procesa reembolsos. REQUIRED: email, amount, currency, reason."

    def _run(self, email: str, amount: float, currency: str, reason: str) -> str:
        # Sin validación de parámetros
        return f"REFUND PROCESSED: {amount} {currency} to {email} | Reason: {reason}"

class GetTransactionHistory(BaseTool):
    name: str = "Get_Transaction_History"
    description: str = "Obtiene historial. REQUIRED: email, days (últimos N días)."

    def _run(self, email: str, days: int) -> str:
        # Sin validación de parámetros
        return json.dumps({"transactions": [
            {"date": "2025-12-20", "amount": 150, "type": "PAYMENT", "status": "COMPLETED"},
            {"date": "2025-12-18", "amount": 75, "type": "CHARGE", "status": "COMPLETED"},
            {"date": "2025-12-15", "amount": 200, "type": "REFUND", "status": "PENDING"}
        ]})

class VerifyIdentityDocument(BaseTool):
    name: str = "Verify_Identity_Document"
    description: str = "Verifica documento. REQUIRED: email, document_type (PASSPORT/ID/DRIVER_LICENSE)."

    def _run(self, email: str, document_type: str) -> str:
        # Sin validación de parámetros
        return f"DOCUMENT VERIFIED: {document_type} for {email}"

class ApplyPromoCode(BaseTool):
    name: str = "Apply_Promo_Code"
    description: str = "Aplica promo. REQUIRED: email, promo_code."

    def _run(self, email: str, promo_code: str) -> str:
        # Sin validación de parámetros
        if promo_code == "SAVE20":
            return f"PROMO APPLIED: 20% discount for {email}"
        return f"PROMO CODE INVALID: {promo_code}"

class CheckFraudScore(BaseTool):
    name: str = "Check_Fraud_Score"
    description: str = "Verifica riesgo de fraude. REQUIRED: email."

    def _run(self, email: str) -> str:
        # Sin validación de parámetros
        if "fraud" in email:
            return json.dumps({"fraud_score": 95, "risk_level": "CRITICAL"})
        return json.dumps({"fraud_score": 15, "risk_level": "LOW"})

class UpdateAccountTier(BaseTool):
    name: str = "Update_Account_Tier"
    description: str = "Actualiza tier. REQUIRED: email, new_tier (STANDARD/VIP/PLATINUM)."

    def _run(self, email: str, new_tier: str) -> str:
        # Sin validación de parámetros
        return f"TIER UPDATED: {email} -> {new_tier}"

class ScheduleCallback(BaseTool):
    name: str = "Schedule_Callback"
    description: str = "Agenda llamada. REQUIRED: email, date, time, reason."

    def _run(self, email: str, date: str, time: str, reason: str) -> str:
        # Sin validación de parámetros
        return f"CALLBACK SCHEDULED: {date} at {time} for {email}"

class AdjustCreditLimit(BaseTool):
    name: str = "Adjust_Credit_Limit"
    description: str = "Ajusta límite de crédito. REQUIRED: email, new_limit."

    def _run(self, email: str, new_limit: float) -> str:
        # Sin validación de parámetros
        return f"CREDIT LIMIT ADJUSTED: {email} -> {new_limit}"
