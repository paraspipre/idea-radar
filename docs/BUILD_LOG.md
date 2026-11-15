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

**Lesson learned the funny way:** I had two AI coding sessions open on the same folder. Both started building M0, and they began overwriting each other's config format mid-run. Rule from now on: one writer per folder. The good ideas from the second session (shared exit codes, context-bound log fields) were merged in by hand.

**Numbers:** 0 LLM calls, 5 CLI commands, 28 tests, strict mypy on the code and the tests.

**Next:** M1, the LLM layer: one `LLMClient` interface, the Agent SDK backend, a content-hash cache, a budget governor, and rate-limit-aware exits.

---

## 2026-11-15: M1, the LLM layer

**What:** one `LLMClient` that every model call in the project must go through. Per call it does:
- a cache lookup (free on a hit),
- a budget check,
- the call itself,
- schema validation,
- a cache write,
- one row in a call log.

Three interchangeable backends sit behind it: the Claude Agent SDK (default, on my Pro subscription), an API stub, and a scripted fake for tests.

**Design choices that save tokens:**
- By default the Agent SDK loads your CLAUDE.md, settings and built-in tool definitions into every call. Pipeline calls turn all of that off and run in an empty sandbox folder with a 25-word system prompt.
- The cache key includes the prompt version, so editing a prompt never mixes old outputs with new ones, and nothing is paid for twice.
- Every retry counts against the budget. Retries cost real usage, so pretending they're free would make the budget a lie.

**Rate limits as a feature:** the SDK emits a `RateLimitEvent` when you approach your plan's limit. On the *warning* (not the hard stop), the run finishes its current call and exits cleanly with code 75. If it does hit the wall, it records when the limit resets, and the next nightly run won't even start before then. Cache hits keep working after a stop because they cost nothing.

**Trust, but log:** each call records `apiKeySource`, which says which credential actually paid. If a call is ever billed to an API key instead of the subscription, it's visible immediately.

**Testing without spending tokens:** 25 new tests. The SDK backend is tested by stubbing `query()` with *real* SDK message objects (results, rate-limit events, error results), so the mapping from 429, 529, 401, refusals and missing output to my error types is pinned down with zero LLM calls.

**Numbers:** 53 tests, 0 LLM calls during development, 1 call needed to verify on the real subscription (`radar llm smoke`).

**Next:** M2, ingestion from Hacker News, Stack Exchange, GitHub Issues and the App Store, with a free prefilter. Still no LLM calls.

---

## 2026-11-15: M2, ingestion and a free prefilter

**What:** four official, free sources, each fetched incrementally from a saved cursor:
- **Hacker News** (Algolia API)
- **Stack Exchange** (`softwarerecs`, `webapps`)
- **GitHub issue search** (feature requests with lots of 👍)
- **App Store reviews** (1–2 stars only)

Every item is cleaned from HTML and PII-scrubbed (emails, phone numbers, @handles, URL query strings), then stored as a short snippet plus a link. Usernames are never stored.

**The prefilter is the token budget's best friend.** Before any LLM sees anything, a regex scorer looks for:
- problem phrases ("I wish there was", "is there a tool", "how do you deal with", "doesn't support")
- willingness-to-pay phrases ("I'd pay", "$20/month", "too expensive")
- engagement, and a per-source prior (a 1-star review is a complaint by construction)
- penalties for hiring posts, self-promotion and spam

**First live run:** 975 items fetched → **66 passed** (7%) → 0 LLM calls. Hacker News is the noisiest source: 27 of 674 new comments passed.

**Small things that mattered:**
- The snippet the LLM will see is a 500-character window *around the first problem signal*, not the start of the post. That way the expensive tokens land on the relevant sentence.
- Dedupe by normalized content hash: the same complaint cross-posted twice counts once.
- I checked each API's real response shape with one request before writing its parser, and the tests replay those shapes through `httpx.MockTransport`.

**Tuning from data, not vibes:** I listed the "near-miss" rejects (score 1.0–1.49). Almost every Stack Exchange one was a genuine software request, so that source's prior went from 1.0 to 1.5. Hacker News near-misses were mixed, so its threshold stays put until the labeled eval (M3) can measure prefilter recall properly.

**Numbers:** 71 tests, 4 live sources, 975 → 66 items, 0 LLM calls.

**Next:** M3, LLM problem extraction (batched, about 25 items per call) plus the hand-labeling tool and the extraction eval.

---

## 2026-11-15: M3, LLM problem extraction plus the eval harness

**What:** stage 2 turns prefilter-passed snippets into structured problems:
- who has it
- the underlying need, in neutral words
- the current workaround
- urgency from 1 to 5
- willingness to pay (none, weak or strong)
- a verbatim quote

**Token design:**
- About **25 items per call**, each a ~500-character snippet.
- The model returns *only the items that are problems*; everything else is omitted instead of echoed, which cuts output tokens.
- Every processed item is marked per extractor version (prompt version + model), so nothing is ever sent twice.
- If the model's JSON doesn't fit the schema, the batch is split in half *once*, then given up on. No infinite retry loops burning quota.
- On my real data: **66 items → 3 calls, about 150 input tokens per item.** `radar extract --dry-run` prints the exact prompt and that estimate without spending anything.

**Anti-hallucination, cheaply:** each extracted problem must carry a short quote, and Python checks that the quote actually appears in the source snippet (ignoring case and punctuation). Unverified quotes are dropped and flagged, so the dashboard only ever shows evidence that exists.

**Evals before vibes:** `radar eval label` is a tiny terminal labeling tool. It samples a *mix* of items the prefilter passed and rejected, because otherwise you can never measure what the free filter throws away. `radar eval extraction` reports:
- **prefilter recall**, which costs nothing,
- **extraction precision, recall and F1** against my labels, which costs about one LLM call per 25 gold items and is cached after that.

**Numbers:** 79 tests, 0 LLM calls during development. The real metrics land as soon as I've labeled about 100 items.

**Next:** M4, local embeddings plus HDBSCAN clustering, batched cluster naming, and demand scores.
