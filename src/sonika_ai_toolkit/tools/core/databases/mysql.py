"""MySQLTool — execute SQL queries against a MySQL/MariaDB database."""

import json
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class _MySQLInput(BaseModel):
    host: str = Field(description="MySQL server hostname.")
    port: int = Field(default=3306, description="MySQL port. Default 3306.")
    database: str = Field(description="Database name.")
    user: str = Field(description="Database user.")
    password: str = Field(description="Database password.")
    query: str = Field(description="SQL statement to execute.")
    params: str = Field(
        default="[]",
        description="JSON array of parameters for a parameterized query.",
    )


class MySQLTool(BaseTool):
    name: str = "mysql_query"
    description: str = (
        "Execute a SQL query against a MySQL or MariaDB database. "
        "Returns rows as JSON for SELECT, or affected row count for INSERT/UPDATE/DELETE."
    )
    args_schema: Type[BaseModel] = _MySQLInput
    risk_hint: int = 1

    def _run(
        self,
        host: str,
        database: str,
        user: str,
        password: str,
        query: str,
        port: int = 3306,
        params: str = "[]",
    ) -> str:
        try:
            import pymysql
            import pymysql.cursors
        except ImportError:
            return "Error: pymysql not installed. Run: pip install pymysql"

        try:
            query_params = json.loads(params)
        except json.JSONDecodeError as e:
            return f"Error: invalid params JSON: {e}"

        try:
            conn = pymysql.connect(
                host=host,
                port=port,
                db=database,
                user=user,
                password=password,
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor,
            )
            try:
                with conn.cursor() as cursor:
                    cursor.execute(query, query_params)
                    stmt = query.strip().upper()
                    if stmt.startswith("SELECT"):
                        rows = cursor.fetchall()
                        return json.dumps(list(rows), ensure_ascii=False, default=str)
                    else:
                        conn.commit()
                        return f"Affected {cursor.rowcount} rows"
            finally:
                conn.close()
        except Exception as e:
            return f"Error: {e}"
