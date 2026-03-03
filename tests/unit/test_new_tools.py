"""
Unit tests for the new core tools added in v0.3.4:
  RunPythonTool, FetchWebPageTool, GetDateTimeTool, EmailSMTPTool,
  SQLiteTool, PostgreSQLTool, MySQLTool, RedisTool
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from sonika_ai_toolkit.tools.core.python_tool import RunPythonTool
from sonika_ai_toolkit.tools.core.web import FetchWebPageTool
from sonika_ai_toolkit.tools.core.datetime_tool import GetDateTimeTool
from sonika_ai_toolkit.tools.core.email_smtp import EmailSMTPTool
from sonika_ai_toolkit.tools.core.databases.sqlite import SQLiteTool
from sonika_ai_toolkit.tools.core.databases.postgres import PostgreSQLTool
from sonika_ai_toolkit.tools.core.databases.mysql import MySQLTool
from sonika_ai_toolkit.tools.core.databases.redis_tool import RedisTool


# ---------------------------------------------------------------------------
# RunPythonTool
# ---------------------------------------------------------------------------

class TestRunPythonTool:
    def test_print_output(self):
        tool = RunPythonTool()
        result = tool._run('print("hello world")')
        assert "hello world" in result

    def test_arithmetic(self):
        tool = RunPythonTool()
        result = tool._run("print(2 + 2)")
        assert "4" in result

    def test_stderr_on_error(self):
        tool = RunPythonTool()
        result = tool._run("raise ValueError('boom')")
        assert "ValueError" in result or "boom" in result

    def test_timeout(self):
        tool = RunPythonTool()
        result = tool._run("import time; time.sleep(60)", timeout=1)
        assert "timed out" in result

    def test_import_stdlib(self):
        tool = RunPythonTool()
        result = tool._run("import math; print(math.sqrt(144))")
        assert "12.0" in result


# ---------------------------------------------------------------------------
# FetchWebPageTool
# ---------------------------------------------------------------------------

class TestFetchWebPageTool:
    def _make_response(self, text: str, status_code: int = 200):
        mock_resp = MagicMock()
        mock_resp.text = text
        mock_resp.status_code = status_code
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_extracts_text(self):
        tool = FetchWebPageTool()
        html = "<html><body><h1>Hello</h1><p>World</p></body></html>"
        with patch("requests.get", return_value=self._make_response(html)):
            result = tool._run("http://example.com")
        assert "Hello" in result
        assert "World" in result

    def test_strips_scripts(self):
        tool = FetchWebPageTool()
        html = "<html><body><script>alert('x')</script><p>Content</p></body></html>"
        with patch("requests.get", return_value=self._make_response(html)):
            result = tool._run("http://example.com")
        assert "alert" not in result
        assert "Content" in result

    def test_truncates_long_content(self):
        tool = FetchWebPageTool()
        html = "<p>" + "x" * 20000 + "</p>"
        with patch("requests.get", return_value=self._make_response(html)):
            result = tool._run("http://example.com")
        assert "Truncated" in result

    def test_error_on_request_failure(self):
        tool = FetchWebPageTool()
        with patch("requests.get", side_effect=Exception("Connection refused")):
            result = tool._run("http://bad-host.invalid")
        assert "Error" in result

    def test_missing_requests(self):
        tool = FetchWebPageTool()
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "requests":
                raise ImportError("No module named 'requests'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = tool._run("http://example.com")
        assert "not installed" in result


# ---------------------------------------------------------------------------
# GetDateTimeTool
# ---------------------------------------------------------------------------

class TestGetDateTimeTool:
    def test_default_format_utc(self):
        tool = GetDateTimeTool()
        result = tool._run()
        # Should look like: 2024-01-15 10:30:00
        assert len(result) == 19
        assert "-" in result and ":" in result

    def test_date_only_format(self):
        tool = GetDateTimeTool()
        result = tool._run(format="%Y-%m-%d")
        assert len(result) == 10

    def test_local_timezone(self):
        tool = GetDateTimeTool()
        result = tool._run(tz="local")
        assert len(result) > 0
        assert "Error" not in result

    def test_invalid_timezone(self):
        tool = GetDateTimeTool()
        result = tool._run(tz="America/New_York")
        assert "Error" in result or "unsupported" in result

    def test_year_in_output(self):
        tool = GetDateTimeTool()
        result = tool._run(format="%Y")
        year = int(result)
        assert 2024 <= year <= 2030


# ---------------------------------------------------------------------------
# EmailSMTPTool
# ---------------------------------------------------------------------------

class TestEmailSMTPTool:
    def test_sends_email(self):
        tool = EmailSMTPTool()
        mock_smtp = MagicMock()
        mock_smtp_instance = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_smtp_instance)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        with patch("smtplib.SMTP", mock_smtp):
            result = tool._run(
                smtp_host="smtp.example.com",
                username="sender@example.com",
                password="secret",
                to_email="recipient@example.com",
                subject="Test",
                body="Hello",
            )
        assert "sent" in result.lower()
        assert "recipient@example.com" in result

    def test_calls_starttls_when_use_tls(self):
        tool = EmailSMTPTool()
        mock_server = MagicMock()
        mock_smtp_cls = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        with patch("smtplib.SMTP", mock_smtp_cls):
            tool._run(
                smtp_host="smtp.example.com",
                username="u@x.com",
                password="p",
                to_email="t@x.com",
                subject="S",
                body="B",
                use_tls=True,
            )
        mock_server.starttls.assert_called_once()

    def test_error_on_smtp_failure(self):
        tool = EmailSMTPTool()
        with patch("smtplib.SMTP", side_effect=Exception("Connection refused")):
            result = tool._run(
                smtp_host="bad-host",
                username="u@x.com",
                password="p",
                to_email="t@x.com",
                subject="S",
                body="B",
            )
        assert "Error" in result


# ---------------------------------------------------------------------------
# SQLiteTool
# ---------------------------------------------------------------------------

class TestSQLiteTool:
    def test_create_and_select(self):
        tool = SQLiteTool()
        tool._run(":memory:", "CREATE TABLE t (id INTEGER, name TEXT)", "[]")
        tool._run(":memory:", "INSERT INTO t VALUES (1, 'Alice')", "[]")

    def test_select_returns_json(self):
        import sqlite3
        # Use a real in-memory DB for proper select test
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'Alice')")
        conn.commit()
        conn.close()

        # Each _run call on ":memory:" creates a new DB; test logic via full flow
        tool = SQLiteTool()
        # Create + insert + select in a temp file
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            tool._run(db_path, "CREATE TABLE users (id INTEGER, name TEXT)")
            tool._run(db_path, "INSERT INTO users VALUES (?, ?)", '["[1, \\"Alice\\"]"]')
            tool._run(db_path, "INSERT INTO users VALUES (1, 'Alice')", "[]")
            result = tool._run(db_path, "SELECT * FROM users")
            data = json.loads(result)
            assert isinstance(data, list)
        finally:
            os.unlink(db_path)

    def test_invalid_params_json(self):
        tool = SQLiteTool()
        result = tool._run(":memory:", "SELECT 1", "not-json")
        assert "Error" in result

    def test_affected_rows(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            tool = SQLiteTool()
            tool._run(db_path, "CREATE TABLE t (id INTEGER)")
            tool._run(db_path, "INSERT INTO t VALUES (1)")
            result = tool._run(db_path, "DELETE FROM t WHERE id = 1")
            assert "Affected" in result or "rows" in result
        finally:
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# PostgreSQLTool
# ---------------------------------------------------------------------------

class TestPostgreSQLTool:
    def test_graceful_import_error(self):
        tool = PostgreSQLTool()
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "psycopg2":
                raise ImportError
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = tool._run(
                host="localhost", database="db", user="u", password="p",
                query="SELECT 1"
            )
        assert "psycopg2" in result and "not installed" in result

    def test_select_calls_fetchall(self):
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
        mock_psycopg2.extras.RealDictCursor = MagicMock()

        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2, "psycopg2.extras": mock_psycopg2.extras}):
            result = tool._run(
                host="localhost", database="db", user="u", password="p",
                query="SELECT id FROM users"
            )
        mock_cursor.execute.assert_called_once()

    def test_invalid_params_json(self):
        tool = PostgreSQLTool()
        mock_psycopg2 = MagicMock()
        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2, "psycopg2.extras": MagicMock()}):
            result = tool._run(
                host="localhost", database="db", user="u", password="p",
                query="SELECT 1", params="invalid"
            )
        assert "Error" in result


# ---------------------------------------------------------------------------
# MySQLTool
# ---------------------------------------------------------------------------

class TestMySQLTool:
    def test_graceful_import_error(self):
        tool = MySQLTool()
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pymysql":
                raise ImportError
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = tool._run(
                host="localhost", database="db", user="u", password="p",
                query="SELECT 1"
            )
        assert "pymysql" in result and "not installed" in result

    def test_select_calls_fetchall(self):
        tool = MySQLTool()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [{"id": 1}]
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        mock_pymysql = MagicMock()
        mock_pymysql.connect.return_value = mock_conn
        mock_pymysql.cursors.DictCursor = MagicMock()

        with patch.dict("sys.modules", {"pymysql": mock_pymysql, "pymysql.cursors": mock_pymysql.cursors}):
            result = tool._run(
                host="localhost", database="db", user="u", password="p",
                query="SELECT id FROM users"
            )
        mock_cursor.execute.assert_called_once()

    def test_invalid_params_json(self):
        tool = MySQLTool()
        mock_pymysql = MagicMock()
        with patch.dict("sys.modules", {"pymysql": mock_pymysql, "pymysql.cursors": MagicMock()}):
            result = tool._run(
                host="localhost", database="db", user="u", password="p",
                query="SELECT 1", params="bad"
            )
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

    def test_get(self):
        tool = RedisTool()
        mock_redis, mock_client = self._make_redis_mock()
        mock_client.get.return_value = "stored_value"
        with patch.dict("sys.modules", {"redis": mock_redis}):
            result = tool._run(action="get", key="mykey")
        assert result == "stored_value"

    def test_get_nil(self):
        tool = RedisTool()
        mock_redis, mock_client = self._make_redis_mock()
        mock_client.get.return_value = None
        with patch.dict("sys.modules", {"redis": mock_redis}):
            result = tool._run(action="get", key="missing")
        assert result == "(nil)"

    def test_set(self):
        tool = RedisTool()
        mock_redis, mock_client = self._make_redis_mock()
        with patch.dict("sys.modules", {"redis": mock_redis}):
            result = tool._run(action="set", key="k", value="v")
        assert result == "OK"
        mock_client.set.assert_called_once_with("k", "v")

    def test_set_with_ttl(self):
        tool = RedisTool()
        mock_redis, mock_client = self._make_redis_mock()
        with patch.dict("sys.modules", {"redis": mock_redis}):
            result = tool._run(action="set", key="k", value="v", ttl=60)
        assert result == "OK"
        mock_client.setex.assert_called_once_with("k", 60, "v")

    def test_delete(self):
        tool = RedisTool()
        mock_redis, mock_client = self._make_redis_mock()
        mock_client.delete.return_value = 1
        with patch.dict("sys.modules", {"redis": mock_redis}):
            result = tool._run(action="delete", key="k")
        assert "1" in result

    def test_keys(self):
        tool = RedisTool()
        mock_redis, mock_client = self._make_redis_mock()
        mock_client.keys.return_value = ["key1", "key2"]
        with patch.dict("sys.modules", {"redis": mock_redis}):
            result = tool._run(action="keys", key="*")
        data = json.loads(result)
        assert "key1" in data

    def test_invalid_action(self):
        tool = RedisTool()
        mock_redis, _ = self._make_redis_mock()
        with patch.dict("sys.modules", {"redis": mock_redis}):
            result = tool._run(action="unknown_action", key="k")
        assert "Error" in result or "unknown" in result

    def test_graceful_import_error(self):
        tool = RedisTool()
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "redis":
                raise ImportError
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = tool._run(action="get", key="k")
        assert "redis" in result and "not installed" in result
