"""PostgreSQLTool — execute SQL queries against a PostgreSQL database."""

import json
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class _PostgreSQLInput(BaseModel):
    host: str = Field(description="PostgreSQL server hostname.")
    port: int = Field(default=5432, description="PostgreSQL port. Default 5432.")
    database: str = Field(description="Database name.")
    user: str = Field(description="Database user.")
    password: str = Field(description="Database password.")
    query: str = Field(description="SQL statement to execute.")
    params: str = Field(
        default="[]",
        description="JSON array of parameters for a parameterized query.",
    )


class PostgreSQLTool(BaseTool):
    name: str = "postgresql_query"
    description: str = (
        "Execute a SQL query against a PostgreSQL database. "
        "Returns rows as JSON for SELECT, or affected row count for INSERT/UPDATE/DELETE."
    )
    args_schema: Type[BaseModel] = _PostgreSQLInput
    risk_hint: int = 1

    def _run(
        self,
        host: str,
        database: str,
        user: str,
        password: str,
        query: str,
        port: int = 5432,
        params: str = "[]",
    ) -> str:
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            return "Error: psycopg2 not installed. Run: pip install psycopg2-binary"

        try:
            query_params = json.loads(params)
        except json.JSONDecodeError as e:
            return f"Error: invalid params JSON: {e}"

        try:
            conn = psycopg2.connect(
                host=host, port=port, dbname=database, user=user, password=password
            )
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(query, query_params)
                    stmt = query.strip().upper()
                    if stmt.startswith("SELECT"):
                        rows = [dict(row) for row in cursor.fetchall()]
                        return json.dumps(rows, ensure_ascii=False, default=str)
                    else:
                        conn.commit()
                        return f"Affected {cursor.rowcount} rows"
            finally:
                conn.close()
        except Exception as e:
            return f"Error: {e}"
