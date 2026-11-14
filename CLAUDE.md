# CLAUDE.md: Idea Radar conventions

Idea Radar finds unmet problems in public discussions, scores demand and competition, sizes each opportunity with sourced numbers, runs a skeptic agent, and ranks the results. It is a single-user personal project, built in public. The full design is in PLAN.md.

## How to work in this repo
- **Plan before code.** PLAN.md is the source of truth. Build **one milestone at a time**. After each one: stop, summarize the changes, and give the exact commands to run (PowerShell syntax; the user is on Windows).
- **Never execute code, run tests, or make LLM calls yourself.** Running things costs the user's limited Claude Pro usage. Write carefully, re-read your own code for bugs, then hand over the commands.
- After every milestone, update **README.md** (keep the mermaid architecture diagram current) and **docs/BUILD_LOG.md** (one dated entry per milestone, written to be posted publicly: what was built, a key decision, a lesson, the numbers).
- Ask at most 5 questions at a time, and only ones that are genuinely blocking. Challenge a default when you have a better option, and give the reason.
- Before using any Claude Agent SDK API, check it against the official docs (code.claude.com/docs/en/agent-sdk) or the installed package. Never guess option names.

## LLM rules (hard constraints)
- Every LLM call goes through `idea_radar.llm.LLMClient`. Nothing else imports `claude_agent_sdk`; only `llm/agent_sdk.py` does.
- The default backend is the Agent SDK on the user's **Pro subscription login**. **Never use or require `ANTHROPIC_API_KEY`.** If it is set while the `agent_sdk` backend is selected, refuse to start. (Pro does not include API credits; the API is billed separately.)
- Pipeline calls always set `setting_sources=[]`, `cwd` to an empty scratch directory, a short custom `system_prompt`, and `tools=[]`, so no CLAUDE.md, settings or tool definitions are billed. Only the competition stage may enable `WebSearch`/`WebFetch`.
- Cheap filtering first. Batch items into each call, use structured output (`output_format` with a pydantic-generated JSON schema), keep schemas small and flat, cap `max_turns`, and keep prompts short.
- Model tiers come from config (`small` for extraction and naming, `strong` for competition, sizing and skeptic). Never hardcode model names in stage code.
- Cache every call by content hash (backend + model + prompt version + schema hash + input). Never reprocess. Prompts live in `llm/prompts/*.vN.md`; bump the version when a prompt changes.
- The budget governor enforces `max_llm_calls` per run and per stage. On `RateLimitEvent` `allowed_warning`, stop after the current call. On `rejected` or a 429, save state and exit with code 75. Every stage must be idempotent and resumable.
- Never invent statistics. Sizing line items cite a stored `market_facts` row (with a URL) or are labeled as assumptions with a confidence level. Python does all the arithmetic.

## Code conventions
- Python 3.12, `uv`, src layout (`src/idea_radar`). Type hints everywhere; mypy (strict on `llm/`, `scoring/` and `pipeline/`); ruff for lint and format.
- pydantic v2 for config and LLM schemas; SQLAlchemy 2.0 + Alembic (it must keep working on Postgres; no SQLite-only SQL outside repository classes).
- structlog JSON logging with `run_id` and `stage` bound. No print statements outside the CLI.
- Configuration comes from YAML in `config/`, with secrets in `.env` (gitignored). No magic numbers in stage code.
- Keep scoring as pure functions in `scoring/` with unit tests. Stage nodes stay thin: load from the DB, call services, write to the DB.
- Tests use `FakeBackend` and fixtures, never the network or a real LLM.
- Data hygiene: store short snippets (≤ 500 characters) and links only; no usernames or PII; `data/` stays gitignored.
- Respect each source's terms of service and rate limits (PLAN.md §10). Use a generic User-Agent with no personal details.
- Git: commit only when the user asks.
