"""MemoryManager — read/write MEMORY.md and SKILLS.md."""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional


class MemoryManager:
    """Manages persistent memory files for an OrchestratorBot instance."""

    MEMORY_FILE = "MEMORY.md"
    SKILLS_FILE = "SKILLS.md"
    SESSIONS_DIR = "sessions"

    # Compression thresholds
    KEEP_FULL_COUNT = 5        # Keep last N sessions in full
    KEEP_SUMMARY_COUNT = 15    # Keep sessions 6-20 as one-liner summaries

    def __init__(self, memory_path: str):
        self.memory_path = memory_path
        self.memory_file = os.path.join(memory_path, self.MEMORY_FILE)
        self.skills_file = os.path.join(memory_path, self.SKILLS_FILE)
        self.sessions_dir = os.path.join(memory_path, self.SESSIONS_DIR)
        self.ensure_dirs()

    # ── Setup ──────────────────────────────────────────────────────────────

    def ensure_dirs(self) -> None:
        """Create memory_path and sessions/ if they don't exist."""
        os.makedirs(self.memory_path, exist_ok=True)
        os.makedirs(self.sessions_dir, exist_ok=True)
        if not os.path.exists(self.memory_file):
            with open(self.memory_file, "w", encoding="utf-8") as f:
                f.write("# OrchestratorBot Memory\n\n")
        if not os.path.exists(self.skills_file):
            with open(self.skills_file, "w", encoding="utf-8") as f:
                f.write("# Dynamic Skills\n\n")

    # ── Reads ──────────────────────────────────────────────────────────────

    def read_memory(self) -> str:
        """Return the full contents of MEMORY.md."""
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    def read_skills(self) -> List[Dict]:
        """Parse SKILLS.md and return a list of skill dicts."""
        skills: List[Dict] = []
        try:
            with open(self.skills_file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return skills

        # Each skill is a JSON object on its own line under a ## header.
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    skills.append(json.loads(line))
                except Exception:
                    pass
        return skills

    # ── Writes ─────────────────────────────────────────────────────────────

    def save_session_log(self, session_id: str, log: List[str]) -> None:
        """Write session log to sessions/{session_id}.md."""
        try:
            path = os.path.join(self.sessions_dir, f"{session_id}.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# Session {session_id}\n\n")
                for line in log:
                    f.write(f"- {line}\n")
        except Exception:
            pass

    def update_memory(
        self,
        session_summary: str,
        new_patterns: Optional[List[str]] = None,
    ) -> None:
        """
        Append session_summary to MEMORY.md and optionally add pattern notes.
        Applies compression: keep last 5 full summaries; drop older ones.
        """
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        entry = f"\n## [{timestamp}]\n{session_summary}\n"
        if new_patterns:
            for p in new_patterns:
                entry += f"- {p}\n"

        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                existing = f.read()
        except Exception:
            existing = "# OrchestratorBot Memory\n\n"

        # Split into header + entries
        sections = existing.split("\n## [")
        header = sections[0]
        old_entries = ["\n## [" + s for s in sections[1:]]

        # Keep only last KEEP_FULL_COUNT entries; drop the rest
        kept = old_entries[-self.KEEP_FULL_COUNT :] if len(old_entries) > self.KEEP_FULL_COUNT else old_entries

        new_content = header + "".join(kept) + entry
        with open(self.memory_file, "w", encoding="utf-8") as f:
            f.write(new_content)

    def add_skill(self, skill_info: Dict) -> None:
        """Append a skill entry (JSON line) to SKILLS.md."""
        try:
            line = json.dumps(skill_info)
            with open(self.skills_file, "a", encoding="utf-8") as f:
                f.write(f"{line}\n")
        except Exception:
            pass
