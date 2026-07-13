"""Unit tests for MemoryManager (orchestrator.memory).

All I/O is confined to tmp_path — no shared state, fully deterministic.
"""

from sonika_ai_toolkit.agents.orchestrator.memory import MemoryManager


class TestSetup:
    def test_ensure_dirs_creates_files_and_sessions(self, tmp_path):
        MemoryManager(str(tmp_path))  # constructor calls ensure_dirs()
        assert (tmp_path / MemoryManager.MEMORY_FILE).exists()
        assert (tmp_path / MemoryManager.SKILLS_FILE).exists()
        assert (tmp_path / MemoryManager.SESSIONS_DIR).is_dir()

    def test_read_memory_returns_seeded_header(self, tmp_path):
        mem = MemoryManager(str(tmp_path))
        assert "OrchestratorBot Memory" in mem.read_memory()


class TestSessionLog:
    def test_save_session_log_writes_file(self, tmp_path):
        mem = MemoryManager(str(tmp_path))
        mem.save_session_log("sess1", ["did a thing", "did another"])
        path = tmp_path / MemoryManager.SESSIONS_DIR / "sess1.md"
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "did a thing" in content
        assert "did another" in content


class TestSkills:
    def test_add_and_read_skill_roundtrip(self, tmp_path):
        mem = MemoryManager(str(tmp_path))
        skill = {"name": "greet", "class": "GreetTool", "file": "greet.py"}
        mem.add_skill(skill)
        skills = mem.read_skills()
        assert skill in skills

    def test_read_skills_empty_when_none(self, tmp_path):
        mem = MemoryManager(str(tmp_path))
        assert mem.read_skills() == []


class TestUpdateMemory:
    def test_update_memory_appends_summary(self, tmp_path):
        mem = MemoryManager(str(tmp_path))
        mem.update_memory("Completed the goal", new_patterns=["retry on 429"])
        content = mem.read_memory()
        assert "Completed the goal" in content
        assert "retry on 429" in content

    def test_update_memory_compresses_old_entries(self, tmp_path):
        mem = MemoryManager(str(tmp_path))
        for i in range(MemoryManager.KEEP_FULL_COUNT + 3):
            mem.update_memory(f"summary-{i}")
        content = mem.read_memory()
        # Oldest entries are dropped; only the last KEEP_FULL_COUNT survive.
        assert "summary-0" not in content
        assert f"summary-{MemoryManager.KEEP_FULL_COUNT + 2}" in content
