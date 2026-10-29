"""Local Flask application for the CyberGuard AI mini project."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from flask import Flask, flash, g, redirect, render_template, request, url_for

from analyzer import analyze_message


APP_DIR = Path(__file__).resolve().parent
INSTANCE_DIR = APP_DIR / "instance"

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("CYBERGUARD_SECRET_KEY", "local-demo-key-change-me")
app.config["DATABASE"] = os.environ.get("CYBERGUARD_DATABASE", str(INSTANCE_DIR / "cyberguard.db"))
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024


@app.template_filter("from_json")
def from_json(value: str) -> object:
    return json.loads(value)


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        database_path = Path(app.config["DATABASE"])
        database_path.parent.mkdir(parents=True, exist_ok=True)
        g.db = sqlite3.connect(database_path)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_error: BaseException | None) -> None:
    database = g.pop("db", None)
    if database is not None:
        database.close()


def initialize_database() -> None:
    database = get_db()
    database.execute(
        """
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            score INTEGER NOT NULL,
            level TEXT NOT NULL,
            evidence TEXT NOT NULL,
            hosts TEXT NOT NULL,
            recommendation TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    database.commit()


@app.before_request
def ensure_database() -> None:
    initialize_database()


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    submitted_message = ""

    if request.method == "POST":
        submitted_message = request.form.get("message", "").strip()
        if not submitted_message:
            flash("Enter a message to analyze.", "error")
        elif len(submitted_message) > 5000:
            flash("Keep the message under 5,000 characters.", "error")
        else:
            database = get_db()
            rows = database.execute("SELECT hosts FROM reports").fetchall()
            previously_seen_hosts = {
                host
                for row in rows
                for host in json.loads(row["hosts"])
            }
            result = analyze_message(submitted_message, previously_seen_hosts)
            database.execute(
                """
                INSERT INTO reports (message, score, level, evidence, hosts, recommendation)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    submitted_message,
                    result["score"],
                    result["level"],
                    json.dumps(result["evidence"]),
                    json.dumps(result["hosts"]),
                    result["recommendation"],
                ),
            )
            database.commit()

    reports = get_db().execute(
        "SELECT id, message, score, level, hosts, created_at FROM reports ORDER BY id DESC LIMIT 10"
    ).fetchall()
    return render_template(
        "index.html",
        result=result,
        submitted_message=submitted_message,
        reports=reports,
    )


@app.post("/clear-history")
def clear_history():
    get_db().execute("DELETE FROM reports")
    get_db().commit()
    flash("Local report history cleared.", "success")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
