import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from app.rules import VerdictResult, evaluate_profile
from app.schemas import ReviewStatus, SystemProfile

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "governance.db"

ALLOWED_TRANSITIONS: dict[ReviewStatus, set[ReviewStatus]] = {
    "draft": {"pending_review", "approved"},
    "pending_review": {"approved", "escalated"},
    "approved": {"escalated"},
    "escalated": {"pending_review"},
}


class GovernanceStore:
    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS submissions (
                    submission_id TEXT PRIMARY KEY,
                    system_name TEXT NOT NULL,
                    profile_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS audit_log (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    submission_id TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    from_status TEXT,
                    to_status TEXT NOT NULL,
                    note TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TRIGGER IF NOT EXISTS prevent_audit_update
                BEFORE UPDATE ON audit_log
                BEGIN
                    SELECT RAISE(ABORT, 'audit_log is append-only: updates are forbidden');
                END;

                CREATE TRIGGER IF NOT EXISTS prevent_audit_delete
                BEFORE DELETE ON audit_log
                BEGIN
                    SELECT RAISE(ABORT, 'audit_log is append-only: deletes are forbidden');
                END;
                """
            )

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def save_intake(
        self,
        system_name: str,
        profile: SystemProfile,
        submission_id: str | None = None,
        actor: str = "interview_agent",
    ) -> tuple[str, VerdictResult]:
        sub_id = submission_id or str(uuid4())[:8]
        verdict_res = evaluate_profile(profile)

        # Force pending_review if deterministic rules found a "No" even if agent missed a flag
        if verdict_res.verdict == "No" and profile.status == "draft":
            profile.status = "pending_review"

        now = self._now()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT status FROM submissions WHERE submission_id = ?", (sub_id,)
            ).fetchone()
            from_status = existing["status"] if existing else None

            conn.execute(
                """
                INSERT INTO submissions (submission_id, system_name, profile_json, status, verdict, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(submission_id) DO UPDATE SET
                    system_name = excluded.system_name,
                    profile_json = excluded.profile_json,
                    status = excluded.status,
                    verdict = excluded.verdict,
                    updated_at = excluded.updated_at
                """,
                (sub_id, system_name, profile.model_dump_json(), profile.status, verdict_res.verdict, now),
            )
            conn.execute(
                """
                INSERT INTO audit_log (submission_id, actor, action, from_status, to_status, note, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (sub_id, actor, "intake_saved", from_status, profile.status, f"Verdict={verdict_res.verdict}", now),
            )
        return sub_id, verdict_res

    def review_submission(
        self,
        submission_id: str,
        new_status: ReviewStatus,
        reviewer: str,
        note: str,
    ) -> None:
        if not note.strip():
            raise ValueError("A non-empty review note is required for auditability.")

        with self._connect() as conn:
            row = conn.execute(
                "SELECT profile_json, status FROM submissions WHERE submission_id = ?",
                (submission_id,),
            ).fetchone()
            if not row:
                raise KeyError(f"Submission '{submission_id}' not found.")

            current_status: ReviewStatus = row["status"]
            if new_status not in ALLOWED_TRANSITIONS[current_status]:
                raise ValueError(f"Invalid transition: {current_status} -> {new_status}")

            profile = SystemProfile.model_validate_json(row["profile_json"])
            profile.status = new_status
            now = self._now()

            conn.execute(
                "UPDATE submissions SET profile_json = ?, status = ?, updated_at = ? WHERE submission_id = ?",
                (profile.model_dump_json(), new_status, now, submission_id),
            )
            conn.execute(
                """
                INSERT INTO audit_log (submission_id, actor, action, from_status, to_status, note, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (submission_id, reviewer, "human_review", current_status, new_status, note.strip(), now),
            )

    def list_submissions(self, status: ReviewStatus | None = None) -> list[dict]:
        with self._connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM submissions WHERE status = ? ORDER BY updated_at DESC", (status,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM submissions ORDER BY updated_at DESC").fetchall()
        return [dict(r) for r in rows]

    def get_audit_log(self, submission_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE submission_id = ? ORDER BY event_id ASC",
                (submission_id,),
            ).fetchall()
        return [dict(r) for r in rows]