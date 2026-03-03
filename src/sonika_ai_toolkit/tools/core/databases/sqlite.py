"""SQLiteTool — execute SQL queries against a SQLite database."""

import json
import sqlite3
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class _SQLiteInput(BaseModel):
    database_path: str = Field(
        description="Path to the SQLite .db file, or ':memory:' for an in-memory database."
    )
    query: str = Field(description="SQL statement to execute (SELECT, INSERT, UPDATE, CREATE, etc.).")
    params: str = Field(
        default="[]",
        description="JSON array of parameters for a parameterized query. E.g. '[\"Alice\", 30]'.",
    )


class SQLiteTool(BaseTool):
    name: str = "sqlite_query"
    description: str = (
        "Execute a SQL query against a SQLite database. "
        "Returns rows as JSON for SELECT, or affected row count for INSERT/UPDATE/DELETE."
    )
    args_schema: Type[BaseModel] = _SQLiteInput
    risk_hint: int = 1

    def _run(self, database_path: str, query: str, params: str = "[]") -> str:
        try:
            query_params = json.loads(params)
        except json.JSONDecodeError as e:
            return f"Error: invalid params JSON: {e}"

        try:
            conn = sqlite3.connect(database_path)
            conn.row_factory = sqlite3.Row
            try:
                cursor = conn.cursor()
                cursor.execute(query, query_params)
                stmt = query.strip().upper()
                if stmt.startswith("SELECT"):
                    rows = [dict(row) for row in cursor.fetchall()]
                    return json.dumps(rows, ensure_ascii=False, default=str)
                else:
                    conn.commit()
                    if cursor.rowcount >= 0:
                        return f"Affected {cursor.rowcount} rows"
                    return "OK"
            finally:
                conn.close()
        except Exception as e:
            return f"Error: {e}"
