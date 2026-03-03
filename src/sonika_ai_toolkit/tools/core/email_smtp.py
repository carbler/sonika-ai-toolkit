"""EmailSMTPTool — sends email via SMTP using stdlib smtplib."""

from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class _EmailSMTPInput(BaseModel):
    smtp_host: str = Field(description="SMTP server hostname (e.g. 'smtp.gmail.com').")
    smtp_port: int = Field(default=587, description="SMTP port. Default 587 (STARTTLS).")
    username: str = Field(description="SMTP username / sender email.")
    password: str = Field(description="SMTP password or app password.")
    to_email: str = Field(description="Recipient email address.")
    subject: str = Field(description="Email subject line.")
    body: str = Field(description="Plain-text email body.")
    use_tls: bool = Field(default=True, description="Use STARTTLS. Default True.")


class EmailSMTPTool(BaseTool):
    name: str = "send_email_smtp"
    description: str = (
        "Send an email via SMTP. Works with Gmail, Outlook, SendGrid SMTP, "
        "or any standard SMTP server. Returns confirmation on success."
    )
    args_schema: Type[BaseModel] = _EmailSMTPInput
    risk_hint: int = 1

    def _run(
        self,
        smtp_host: str,
        username: str,
        password: str,
        to_email: str,
        subject: str,
        body: str,
        smtp_port: int = 587,
        use_tls: bool = True,
    ) -> str:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        try:
            msg = MIMEMultipart()
            msg["From"] = username
            msg["To"] = to_email
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
                if use_tls:
                    server.starttls()
                server.login(username, password)
                server.sendmail(username, to_email, msg.as_string())

            return f"Email sent to {to_email}"
        except Exception as e:
            return f"Error sending email: {e}"
