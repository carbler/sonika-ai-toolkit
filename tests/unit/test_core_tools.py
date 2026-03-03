"""
Functional unit tests for all core tools except database tools.

Covers: RunBashTool, BashSafeTool, ReadFileTool, WriteFileTool,
        ListDirTool, DeleteFileTool, FindFileTool, CallApiTool,
        SearchWebTool, RunPythonTool, FetchWebPageTool,
        GetDateTimeTool, EmailSMTPTool.
"""

import json
import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from sonika_ai_toolkit.tools.core.bash import BashSafeTool, RunBashTool
from sonika_ai_toolkit.tools.core.datetime_tool import GetDateTimeTool
from sonika_ai_toolkit.tools.core.email_smtp import EmailSMTPTool
from sonika_ai_toolkit.tools.core.files import (
    DeleteFileTool,
    FindFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from sonika_ai_toolkit.tools.core.http import CallApiTool
from sonika_ai_toolkit.tools.core.python_tool import RunPythonTool
from sonika_ai_toolkit.tools.core.search import SearchWebTool
from sonika_ai_toolkit.tools.core.web import FetchWebPageTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path):
    """Pytest tmp_path as string."""
    return str(tmp_path)


@pytest.fixture
def tmp_file(tmp_path):
    """A real temporary file with known content."""
    f = tmp_path / "sample.txt"
    f.write_text("hello sonika", encoding="utf-8")
    return str(f)


# ---------------------------------------------------------------------------
# RunBashTool
# ---------------------------------------------------------------------------

class TestRunBashTool:
    def test_stdout_captured(self):
        result = RunBashTool()._run("echo hello")
        assert "hello" in result

    def test_stderr_captured(self):
        result = RunBashTool()._run("echo error_msg >&2")
        assert "error_msg" in result

    def test_no_output_shows_exit_code(self):
        result = RunBashTool()._run("true")
        assert "exit code" in result or result == ""  # empty stdout/stderr is possible

    def test_timeout_triggers(self):
        result = RunBashTool()._run("sleep 60", timeout=1)
        assert "timed out" in result

    def test_multiline_output(self):
        result = RunBashTool()._run("printf 'a\\nb\\nc'")
        assert "a" in result and "b" in result and "c" in result

    def test_nonzero_exit_still_returns_output(self):
        result = RunBashTool()._run("ls /nonexistent_path_xyz 2>&1; true")
        assert isinstance(result, str)

    def test_pipeline(self):
        result = RunBashTool()._run("echo hello | tr a-z A-Z")
        assert "HELLO" in result

    def test_env_variable(self):
        result = RunBashTool()._run("MY_VAR=sonika && echo $MY_VAR")
        assert "sonika" in result


# ---------------------------------------------------------------------------
# BashSafeTool
# ---------------------------------------------------------------------------

class TestBashSafeTool:
    def test_echo_allowed(self):
        result = BashSafeTool()._run("echo safe")
        assert "safe" in result

    def test_rm_blocked(self):
        result = BashSafeTool()._run("rm /tmp/something")
        assert "forbidden" in result.lower() or "ERROR" in result

    def test_sudo_blocked(self):
        result = BashSafeTool()._run("sudo ls")
        assert "forbidden" in result.lower() or "ERROR" in result

    def test_mv_blocked(self):
        result = BashSafeTool()._run("mv /tmp/a /tmp/b")
        assert "forbidden" in result.lower() or "ERROR" in result

    def test_dd_blocked(self):
        result = BashSafeTool()._run("dd if=/dev/zero")
        assert "forbidden" in result.lower() or "ERROR" in result

    def test_mkfs_blocked(self):
        result = BashSafeTool()._run("mkfs /dev/sda")
        assert "forbidden" in result.lower() or "ERROR" in result

    def test_ls_allowed(self):
        result = BashSafeTool()._run("ls /tmp")
        assert "ERROR" not in result or "forbidden" not in result.lower()

    def test_cat_allowed(self):
        result = BashSafeTool()._run("echo test | cat")
        assert "test" in result


# ---------------------------------------------------------------------------
# ReadFileTool
# ---------------------------------------------------------------------------

class TestReadFileTool:
    def test_reads_existing_file(self, tmp_file):
        result = ReadFileTool()._run(tmp_file)
        assert result == "hello sonika"

    def test_error_on_missing_file(self):
        result = ReadFileTool()._run("/nonexistent/path/file.txt")
        assert "Error" in result

    def test_reads_unicode(self, tmp_path):
        f = tmp_path / "unicode.txt"
        f.write_text("café ñoño 日本語", encoding="utf-8")
        result = ReadFileTool()._run(str(f))
        assert "café" in result
        assert "日本語" in result

    def test_reads_multiline(self, tmp_path):
        f = tmp_path / "multi.txt"
        f.write_text("line1\nline2\nline3", encoding="utf-8")
        result = ReadFileTool()._run(str(f))
        assert "line1" in result
        assert "line3" in result


# ---------------------------------------------------------------------------
# WriteFileTool
# ---------------------------------------------------------------------------

class TestWriteFileTool:
    def test_creates_file(self, tmp_dir):
        path = os.path.join(tmp_dir, "new.txt")
        result = WriteFileTool()._run(path, "content here")
        assert os.path.exists(path)
        assert "Written" in result

    def test_reports_char_count(self, tmp_dir):
        path = os.path.join(tmp_dir, "counted.txt")
        content = "x" * 100
        result = WriteFileTool()._run(path, content)
        assert "100" in result

    def test_creates_parent_dirs(self, tmp_dir):
        path = os.path.join(tmp_dir, "a", "b", "c", "deep.txt")
        WriteFileTool()._run(path, "deep content")
        assert os.path.exists(path)

    def test_content_roundtrip(self, tmp_dir):
        path = os.path.join(tmp_dir, "rt.txt")
        WriteFileTool()._run(path, "roundtrip test")
        assert open(path).read() == "roundtrip test"

    def test_overwrites_existing(self, tmp_file):
        WriteFileTool()._run(tmp_file, "new content")
        assert open(tmp_file).read() == "new content"


# ---------------------------------------------------------------------------
# ListDirTool
# ---------------------------------------------------------------------------

class TestListDirTool:
    def test_lists_files(self, tmp_path):
        (tmp_path / "file1.txt").write_text("a")
        (tmp_path / "file2.txt").write_text("b")
        result = ListDirTool()._run(str(tmp_path))
        assert "file1.txt" in result
        assert "file2.txt" in result

    def test_tags_files_and_dirs(self, tmp_path):
        (tmp_path / "myfile.txt").write_text("x")
        (tmp_path / "subdir").mkdir()
        result = ListDirTool()._run(str(tmp_path))
        assert "[file]" in result
        assert "[dir]" in result

    def test_empty_dir_message(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = ListDirTool()._run(str(empty))
        assert "empty" in result.lower()

    def test_error_on_missing_dir(self):
        result = ListDirTool()._run("/nonexistent/dir/xyz")
        assert "Error" in result

    def test_sorted_output(self, tmp_path):
        for name in ["zzz.txt", "aaa.txt", "mmm.txt"]:
            (tmp_path / name).write_text("")
        result = ListDirTool()._run(str(tmp_path))
        lines = [l for l in result.splitlines() if l.strip()]
        names = [l.split()[-1] for l in lines]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# DeleteFileTool
# ---------------------------------------------------------------------------

class TestDeleteFileTool:
    def test_deletes_file(self, tmp_file):
        result = DeleteFileTool()._run(tmp_file)
        assert not os.path.exists(tmp_file)
        assert "Deleted" in result

    def test_error_on_missing_file(self):
        result = DeleteFileTool()._run("/nonexistent/file.txt")
        assert "Error" in result

    def test_result_contains_path(self, tmp_file):
        result = DeleteFileTool()._run(tmp_file)
        assert tmp_file in result


# ---------------------------------------------------------------------------
# FindFileTool
# ---------------------------------------------------------------------------

class TestFindFileTool:
    def test_finds_by_exact_name(self, tmp_path):
        (tmp_path / "target.txt").write_text("x")
        result = FindFileTool()._run("target.txt", str(tmp_path))
        assert "target.txt" in result

    def test_finds_by_glob(self, tmp_path):
        (tmp_path / "data1.csv").write_text("")
        (tmp_path / "data2.csv").write_text("")
        (tmp_path / "notes.txt").write_text("")
        result = FindFileTool()._run("*.csv", str(tmp_path))
        assert "data1.csv" in result
        assert "data2.csv" in result
        assert "notes.txt" not in result

    def test_case_insensitive(self, tmp_path):
        (tmp_path / "README.MD").write_text("")
        result = FindFileTool()._run("readme.md", str(tmp_path))
        assert "README.MD" in result

    def test_recurses_subdirs(self, tmp_path):
        sub = tmp_path / "sub" / "deep"
        sub.mkdir(parents=True)
        (sub / "hidden.txt").write_text("")
        result = FindFileTool()._run("hidden.txt", str(tmp_path))
        assert "hidden.txt" in result

    def test_no_match_message(self, tmp_path):
        result = FindFileTool()._run("doesnotexist.xyz", str(tmp_path))
        assert "No files found" in result


# ---------------------------------------------------------------------------
# CallApiTool
# ---------------------------------------------------------------------------

class TestCallApiTool:
    def _mock_response(self, json_data=None, text="", status_code=200):
        mock = MagicMock()
        mock.status_code = status_code
        if json_data is not None:
            mock.json.return_value = json_data
        else:
            mock.json.side_effect = Exception("not json")
        mock.text = text
        return mock

    def test_get_returns_status(self):
        resp = self._mock_response(json_data={"ok": True})
        with patch("requests.request", return_value=resp):
            result = CallApiTool()._run("GET", "http://api.example.com/data")
        assert "200" in result

    def test_get_returns_json_body(self):
        resp = self._mock_response(json_data={"key": "value"})
        with patch("requests.request", return_value=resp):
            result = CallApiTool()._run("GET", "http://api.example.com/data")
        assert "value" in result

    def test_post_passes_body(self):
        resp = self._mock_response(json_data={"created": True})
        with patch("requests.request", return_value=resp) as mock_req:
            CallApiTool()._run("POST", "http://api.example.com/items", body={"name": "x"})
        _, kwargs = mock_req.call_args
        assert kwargs.get("json") == {"name": "x"}

    def test_non_json_response(self):
        resp = self._mock_response(text="plain text response")
        with patch("requests.request", return_value=resp):
            result = CallApiTool()._run("GET", "http://example.com")
        assert "plain text response" in result

    def test_passes_headers(self):
        resp = self._mock_response(json_data={})
        with patch("requests.request", return_value=resp) as mock_req:
            CallApiTool()._run("GET", "http://api.example.com", headers={"Authorization": "Bearer token"})
        _, kwargs = mock_req.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer token"

    def test_error_on_exception(self):
        with patch("requests.request", side_effect=Exception("timeout")):
            result = CallApiTool()._run("GET", "http://bad.host")
        assert "Error" in result

    def test_method_uppercased(self):
        resp = self._mock_response(json_data={})
        with patch("requests.request", return_value=resp) as mock_req:
            CallApiTool()._run("get", "http://api.example.com")
        args, _ = mock_req.call_args
        assert args[0] == "GET"


# ---------------------------------------------------------------------------
# SearchWebTool
# ---------------------------------------------------------------------------

class TestSearchWebTool:
    def test_stub_when_no_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("SERPER_API_KEY", None)
            os.environ.pop("SERPAPI_API_KEY", None)
            result = SearchWebTool()._run("python tutorial")
        assert "stub" in result.lower() or "no api key" in result.lower() or "not configured" in result.lower()

    def test_stub_includes_query(self):
        env = {k: v for k, v in os.environ.items() if k not in ("SERPER_API_KEY", "SERPAPI_API_KEY")}
        with patch.dict(os.environ, env, clear=True):
            result = SearchWebTool()._run("my search query")
        assert "my search query" in result

    def test_serper_called_when_key_present(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "organic": [
                {"title": "Result 1", "snippet": "Snippet 1", "link": "http://r1.com"},
            ]
        }
        with patch.dict(os.environ, {"SERPER_API_KEY": "fake-key"}):
            with patch("requests.post", return_value=mock_resp):
                result = SearchWebTool()._run("python")
        assert "Result 1" in result
        assert "Snippet 1" in result

    def test_serper_no_results(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"organic": []}
        with patch.dict(os.environ, {"SERPER_API_KEY": "fake-key"}):
            with patch("requests.post", return_value=mock_resp):
                result = SearchWebTool()._run("xyzzy12345")
        assert "No results" in result

    def test_serper_error_handled(self):
        with patch.dict(os.environ, {"SERPER_API_KEY": "fake-key"}):
            with patch("requests.post", side_effect=Exception("network error")):
                result = SearchWebTool()._run("query")
        assert "error" in result.lower()


# ---------------------------------------------------------------------------
# RunPythonTool
# ---------------------------------------------------------------------------

class TestRunPythonTool:
    def test_print_stdout(self):
        result = RunPythonTool()._run('print("hello")')
        assert "hello" in result

    def test_arithmetic(self):
        result = RunPythonTool()._run("print(6 * 7)")
        assert "42" in result

    def test_multiline_code(self):
        code = "total = sum(range(10))\nprint(total)"
        result = RunPythonTool()._run(code)
        assert "45" in result

    def test_import_math(self):
        result = RunPythonTool()._run("import math; print(round(math.pi, 4))")
        assert "3.1416" in result

    def test_exception_captured_in_stderr(self):
        result = RunPythonTool()._run("raise RuntimeError('oops')")
        assert "RuntimeError" in result or "oops" in result

    def test_syntax_error_captured(self):
        result = RunPythonTool()._run("def bad(: pass")
        assert "Error" in result or "Syntax" in result

    def test_timeout(self):
        result = RunPythonTool()._run("import time; time.sleep(60)", timeout=1)
        assert "timed out" in result

    def test_json_output(self):
        result = RunPythonTool()._run("import json; print(json.dumps({'x': 1}))")
        assert '"x"' in result


# ---------------------------------------------------------------------------
# FetchWebPageTool
# ---------------------------------------------------------------------------

class TestFetchWebPageTool:
    def _resp(self, html, status=200):
        m = MagicMock()
        m.text = html
        m.status_code = status
        m.raise_for_status = MagicMock()
        return m

    def test_extracts_paragraph_text(self):
        with patch("requests.get", return_value=self._resp("<p>Hello world</p>")):
            assert "Hello world" in FetchWebPageTool()._run("http://x.com")

    def test_strips_script_tags(self):
        html = "<script>alert('xss')</script><p>Clean</p>"
        with patch("requests.get", return_value=self._resp(html)):
            result = FetchWebPageTool()._run("http://x.com")
        assert "alert" not in result
        assert "Clean" in result

    def test_strips_style_tags(self):
        html = "<style>body { color: red }</style><p>Text</p>"
        with patch("requests.get", return_value=self._resp(html)):
            result = FetchWebPageTool()._run("http://x.com")
        assert "color" not in result
        assert "Text" in result

    def test_decodes_html_entities(self):
        html = "<p>&amp; &lt; &gt; &nbsp; &quot;</p>"
        with patch("requests.get", return_value=self._resp(html)):
            result = FetchWebPageTool()._run("http://x.com")
        assert "&" in result
        assert "<" in result or ">" in result

    def test_truncates_large_page(self):
        html = "<p>" + "word " * 5000 + "</p>"
        with patch("requests.get", return_value=self._resp(html)):
            result = FetchWebPageTool()._run("http://x.com")
        assert "Truncated" in result

    def test_error_on_network_failure(self):
        with patch("requests.get", side_effect=ConnectionError("refused")):
            result = FetchWebPageTool()._run("http://bad.host")
        assert "Error" in result

    def test_empty_page(self):
        with patch("requests.get", return_value=self._resp("   ")):
            result = FetchWebPageTool()._run("http://x.com")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# GetDateTimeTool
# ---------------------------------------------------------------------------

class TestGetDateTimeTool:
    def test_default_utc_format(self):
        result = GetDateTimeTool()._run()
        # Should match: 2024-01-15 10:30:00
        assert len(result) == 19
        parts = result.split(" ")
        assert len(parts) == 2
        date_parts = parts[0].split("-")
        assert len(date_parts) == 3

    def test_year_is_reasonable(self):
        result = GetDateTimeTool()._run(format="%Y")
        assert 2024 <= int(result) <= 2030

    def test_date_only_format(self):
        result = GetDateTimeTool()._run(format="%Y-%m-%d")
        assert len(result) == 10
        assert result.count("-") == 2

    def test_time_only_format(self):
        result = GetDateTimeTool()._run(format="%H:%M:%S")
        assert len(result) == 8
        assert result.count(":") == 2

    def test_local_tz(self):
        result = GetDateTimeTool()._run(tz="local")
        assert "Error" not in result
        assert len(result) > 0

    def test_utc_tz_explicit(self):
        result = GetDateTimeTool()._run(tz="UTC")
        assert "Error" not in result

    def test_invalid_tz_returns_error(self):
        result = GetDateTimeTool()._run(tz="Mars/Olympus")
        assert "Error" in result or "unsupported" in result

    def test_unix_timestamp_format(self):
        result = GetDateTimeTool()._run(format="%s")
        # %s might not work on all platforms, just check it returns something
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# EmailSMTPTool
# ---------------------------------------------------------------------------

class TestEmailSMTPTool:
    def _smtp_ctx(self):
        mock_server = MagicMock()
        mock_cls = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        return mock_cls, mock_server

    def test_success_message(self):
        mock_cls, _ = self._smtp_ctx()
        with patch("smtplib.SMTP", mock_cls):
            result = EmailSMTPTool()._run(
                smtp_host="smtp.example.com", username="u@x.com",
                password="p", to_email="t@x.com", subject="S", body="B",
            )
        assert "sent" in result.lower()

    def test_recipient_in_result(self):
        mock_cls, _ = self._smtp_ctx()
        with patch("smtplib.SMTP", mock_cls):
            result = EmailSMTPTool()._run(
                smtp_host="smtp.example.com", username="u@x.com",
                password="p", to_email="recipient@domain.com",
                subject="S", body="B",
            )
        assert "recipient@domain.com" in result

    def test_starttls_called_when_tls_true(self):
        mock_cls, mock_server = self._smtp_ctx()
        with patch("smtplib.SMTP", mock_cls):
            EmailSMTPTool()._run(
                smtp_host="smtp.example.com", username="u@x.com",
                password="p", to_email="t@x.com", subject="S",
                body="B", use_tls=True,
            )
        mock_server.starttls.assert_called_once()

    def test_starttls_not_called_when_tls_false(self):
        mock_cls, mock_server = self._smtp_ctx()
        with patch("smtplib.SMTP", mock_cls):
            EmailSMTPTool()._run(
                smtp_host="smtp.example.com", username="u@x.com",
                password="p", to_email="t@x.com", subject="S",
                body="B", use_tls=False,
            )
        mock_server.starttls.assert_not_called()

    def test_login_called_with_credentials(self):
        mock_cls, mock_server = self._smtp_ctx()
        with patch("smtplib.SMTP", mock_cls):
            EmailSMTPTool()._run(
                smtp_host="smtp.example.com", username="myuser@x.com",
                password="mypassword", to_email="t@x.com",
                subject="S", body="B",
            )
        mock_server.login.assert_called_once_with("myuser@x.com", "mypassword")

    def test_sendmail_called(self):
        mock_cls, mock_server = self._smtp_ctx()
        with patch("smtplib.SMTP", mock_cls):
            EmailSMTPTool()._run(
                smtp_host="smtp.example.com", username="u@x.com",
                password="p", to_email="dest@x.com",
                subject="S", body="B",
            )
        mock_server.sendmail.assert_called_once()

    def test_error_on_connection_failure(self):
        with patch("smtplib.SMTP", side_effect=ConnectionRefusedError("refused")):
            result = EmailSMTPTool()._run(
                smtp_host="bad-host", username="u@x.com",
                password="p", to_email="t@x.com", subject="S", body="B",
            )
        assert "Error" in result

    def test_smtp_port_passed(self):
        mock_cls, _ = self._smtp_ctx()
        with patch("smtplib.SMTP", mock_cls):
            EmailSMTPTool()._run(
                smtp_host="smtp.example.com", smtp_port=465,
                username="u@x.com", password="p",
                to_email="t@x.com", subject="S", body="B",
            )
        args, _ = mock_cls.call_args
        assert args[1] == 465
