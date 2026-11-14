# Idea Radar: Build Log

A public log of building Idea Radar: an agentic pipeline that scans the internet for unmet problems and ranks them as business opportunities, with evidence for every claim.

---

## 2026-11-15: Day 0, the plan

**What:** Wrote the architecture before writing any code. The pipeline has 8 stages: ingest, LLM problem extraction, local-embedding clustering, demand scoring, competition research, opportunity sizing, a skeptic agent that tries to kill each idea, and transparent ranking. Stages 2 to 8 are orchestrated as a LangGraph graph.

**The constraint that shapes everything:** the whole project runs on one Claude Pro subscription through the Claude Agent SDK. There is no API budget, so token efficiency is a design requirement:
- Free filters first (regex, score thresholds, dedupe), so most items never reach an LLM.
- About 25 items per extraction call, with structured JSON output.
- A content-hash cache, so nothing is processed twice.
- A small model for extraction and a stronger one only for the 3 best candidates each night.
- A budget governor that stops cleanly at a rate-limit warning and resumes the next night.

**Decisions:**
- Sizing math is done by Python, not the LLM. The model picks the inputs, and each input must cite a source URL or be labeled as an assumption, so the totals can't be hallucinated.
- One finding from the docs: the Agent SDK loads CLAUDE.md and settings into every call by default. Turning that off for pipeline calls saves tokens on every request.
- Reddit now requires approval for new API access, so Stack Exchange and GitHub Issues were added as official sources.

**Next:** M0, the project skeleton (config, database, CLI, `radar doctor`).

---

## 2026-11-15: M0, the skeleton

**What:** the boring foundation that keeps everything else honest:
- **Layered config:** YAML defaults, a local override file, then `RADAR__*` env vars, all validated by pydantic. A typo in a key fails loudly instead of being silently ignored.
- **Database:** SQLAlchemy 2.0 with Alembic migrations on SQLite, written so the same code moves to Postgres later. The first migration has only a `runs` table; each milestone adds its own tables in its own migration.
- **CLI:** a Typer command, `radar`, with `doctor`, `config`, `db init` and `db status`.
- **CI:** GitHub Actions running ruff, strict mypy, and pytest. The tests never touch the network or an LLM.

**Key decision: a guard against surprise bills.** The Claude Agent SDK prefers an `ANTHROPIC_API_KEY` from the environment over your subscription login. One stray env var (or a line in `.env`) would quietly move every call onto API billing. So `radar doctor` checks for it, and the LLM layer will refuse to run if it's set. There's a test that plants a key in `.env` and expects doctor to catch it.

**Small things I'm glad I did:**
- A test that diffs the ORM models against the migrations, so they can never drift apart.
- Timezone-aware UTC datetimes enforced at the column type, because SQLite silently drops time zones.
- Doctor warns when the SQLite file sits inside OneDrive, since cloud sync and SQLite don't mix.

**Numbers:** 0 LLM calls, 5 CLI commands, 28 tests.

**Next:** M1, the LLM layer: one `LLMClient` interface, the Agent SDK backend, a content-hash cache, a budget governor, and rate-limit-aware exits.
