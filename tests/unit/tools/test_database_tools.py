"""
Unit tests for the database tools:
  SQLiteTool, PostgreSQLTool, MySQLTool, RedisTool.

SQLite uses real temp files (fast, no server). Postgres/MySQL/Redis mock the
underlying driver so no live server is required.
"""

import json
from unittest.mock import MagicMock, patch

from sonika_ai_toolkit.tools.core.databases.sqlite import SQLiteTool
from sonika_ai_toolkit.tools.core.databases.postgres import PostgreSQLTool
from sonika_ai_toolkit.tools.core.databases.mysql import MySQLTool
from sonika_ai_toolkit.tools.core.databases.redis_tool import RedisTool


# ---------------------------------------------------------------------------
# SQLiteTool
# ---------------------------------------------------------------------------

class TestSQLiteTool:
    def test_create_and_select_roundtrip(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        tool = SQLiteTool()
        tool._run(db_path, "CREATE TABLE users (id INTEGER, name TEXT)")
        tool._run(db_path, "INSERT INTO users VALUES (1, 'Alice')", "[]")
        result = tool._run(db_path, "SELECT * FROM users")
        data = json.loads(result)
        assert isinstance(data, list)
        assert data and data[0].get("name") == "Alice"

    def test_invalid_params_json(self):
        result = SQLiteTool()._run(":memory:", "SELECT 1", "not-json")
        assert "Error" in result

    def test_affected_rows_reported(self, tmp_path):
        db_path = str(tmp_path / "t.db")
        tool = SQLiteTool()
        tool._run(db_path, "CREATE TABLE t (id INTEGER)")
        tool._run(db_path, "INSERT INTO t VALUES (1)")
        result = tool._run(db_path, "DELETE FROM t WHERE id = 1")
        assert "Affected" in result or "rows" in result


# ---------------------------------------------------------------------------
# PostgreSQLTool
# ---------------------------------------------------------------------------

class TestPostgreSQLTool:
    def test_graceful_import_error(self):
        tool = PostgreSQLTool()
        real_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "psycopg2":
                raise ImportError
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = tool._run(host="localhost", database="db", user="u",
                               password="p", query="SELECT 1")
        assert "psycopg2" in result and "not installed" in result

    def test_select_executes_query(self):
        tool = PostgreSQLTool()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [{"id": 1}]
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        mock_psycopg2 = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn

        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2,
                                        "psycopg2.extras": mock_psycopg2.extras}):
            tool._run(host="localhost", database="db", user="u",
                      password="p", query="SELECT id FROM users")
        mock_cursor.execute.assert_called_once()

    def test_invalid_params_json(self):
        tool = PostgreSQLTool()
        mock_psycopg2 = MagicMock()
        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2,
                                        "psycopg2.extras": MagicMock()}):
            result = tool._run(host="localhost", database="db", user="u",
                               password="p", query="SELECT 1", params="invalid")
        assert "Error" in result


# ---------------------------------------------------------------------------
# MySQLTool
# ---------------------------------------------------------------------------

class TestMySQLTool:
    def test_graceful_import_error(self):
        tool = MySQLTool()
        real_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "pymysql":
                raise ImportError
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = tool._run(host="localhost", database="db", user="u",
                               password="p", query="SELECT 1")
        assert "pymysql" in result and "not installed" in result

    def test_select_executes_query(self):
        tool = MySQLTool()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [{"id": 1}]
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        mock_pymysql = MagicMock()
        mock_pymysql.connect.return_value = mock_conn

        with patch.dict("sys.modules", {"pymysql": mock_pymysql,
                                        "pymysql.cursors": mock_pymysql.cursors}):
            tool._run(host="localhost", database="db", user="u",
                      password="p", query="SELECT id FROM users")
        mock_cursor.execute.assert_called_once()

    def test_invalid_params_json(self):
        tool = MySQLTool()
        mock_pymysql = MagicMock()
        with patch.dict("sys.modules", {"pymysql": mock_pymysql,
                                        "pymysql.cursors": MagicMock()}):
            result = tool._run(host="localhost", database="db", user="u",
                               password="p", query="SELECT 1", params="bad")
        assert "Error" in result


# ---------------------------------------------------------------------------
# RedisTool
# ---------------------------------------------------------------------------

class TestRedisTool:
    def _make_redis_mock(self):
        mock_redis_module = MagicMock()
        mock_client = MagicMock()
        mock_redis_module.Redis.return_value = mock_client
        return mock_redis_module, mock_client

    def test_get_returns_value(self):
        mock_redis, mock_client = self._make_redis_mock()
        mock_client.get.return_value = "stored_value"
        with patch.dict("sys.modules", {"redis": mock_redis}):
            result = RedisTool()._run(action="get", key="mykey")
        assert result == "stored_value"

    def test_get_missing_returns_nil(self):
        mock_redis, mock_client = self._make_redis_mock()
        mock_client.get.return_value = None
        with patch.dict("sys.modules", {"redis": mock_redis}):
            result = RedisTool()._run(action="get", key="missing")
        assert result == "(nil)"

    def test_set(self):
        mock_redis, mock_client = self._make_redis_mock()
        with patch.dict("sys.modules", {"redis": mock_redis}):
            result = RedisTool()._run(action="set", key="k", value="v")
        assert result == "OK"
        mock_client.set.assert_called_once_with("k", "v")

    def test_set_with_ttl_uses_setex(self):
        mock_redis, mock_client = self._make_redis_mock()
        with patch.dict("sys.modules", {"redis": mock_redis}):
            RedisTool()._run(action="set", key="k", value="v", ttl=60)
        mock_client.setex.assert_called_once_with("k", 60, "v")

    def test_delete(self):
        mock_redis, mock_client = self._make_redis_mock()
        mock_client.delete.return_value = 1
        with patch.dict("sys.modules", {"redis": mock_redis}):
            result = RedisTool()._run(action="delete", key="k")
        assert "1" in result

    def test_keys(self):
        mock_redis, mock_client = self._make_redis_mock()
        mock_client.keys.return_value = ["key1", "key2"]
        with patch.dict("sys.modules", {"redis": mock_redis}):
            result = RedisTool()._run(action="keys", key="*")
        assert "key1" in json.loads(result)

    def test_invalid_action(self):
        mock_redis, _ = self._make_redis_mock()
        with patch.dict("sys.modules", {"redis": mock_redis}):
            result = RedisTool()._run(action="unknown_action", key="k")
        assert "Error" in result or "unknown" in result

    def test_graceful_import_error(self):
        real_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "redis":
                raise ImportError
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = RedisTool()._run(action="get", key="k")
        assert "redis" in result and "not installed" in result
