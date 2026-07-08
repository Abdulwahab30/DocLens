#!/usr/bin/env python3
"""
eval/run_eval.py  —  DocLens Retrieval Evaluation  (Hit-Rate @ K + Faithfulness)
==================================================================

Loads questions.json, fires each question at DocLens, checks whether
the expected source_page appears in the top-K retrieved chunk citations,
and prints a per-question log plus a final hit_rate summary.

Optionally (--judge), also runs an LLM-as-judge faithfulness check: it shows
the judge the assistant's generated answer plus the expected_answer and asks
whether the assistant's answer is factually consistent with it. This catches
cases retrieval hit-rate can't: right page retrieved but a hallucinated or
wrong answer synthesized from it.

Two operation modes
───────────────────
  --mode api     (default) Calls the running FastAPI server.
                 Requires --email / --password
                 (or env vars DOCLENS_EMAIL, DOCLENS_PASSWORD).
                 Retrieval is scoped to all of that user's ready documents —
                 there is no per-document filter in the API.

  --mode direct  Imports backend.app.services.retrieval directly.
                 Run from the repo root so the import path resolves:
                     python eval/run_eval.py --mode direct --user-id <uuid>

Usage examples
──────────────
  # API mode (server must be running on localhost:8000)
  python eval/run_eval.py \\
      --email admin@example.com \\
      --password secret

  # Custom questions file and top-k
  python eval/run_eval.py --questions eval/questions.json --top-k 3

  # Direct mode (no server needed)
  python eval/run_eval.py --mode direct --user-id <uuid>

  # Pipe JSON results to a file
  python eval/run_eval.py --json-out eval/results.json

  # Also run the LLM faithfulness judge (needs OPENROUTER_API_KEY)
  python eval/run_eval.py --email a@b.com --password secret --judge
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ── ANSI colour helpers (no third-party deps) ─────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
WHITE  = "\033[97m"


def _c(text: str, *codes: str) -> str:
    """Wrap text in ANSI codes (skipped when stdout is not a tty)."""
    if not sys.stdout.isatty():
        return str(text)
    return "".join(codes) + str(text) + RESET


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Question:
    id: int
    question: str
    expected_answer: str
    source_page: int


@dataclass
class QuestionResult:
    id: int
    question: str
    source_page: int
    retrieved_pages: list[int]
    hit: bool
    latency_ms: float
    error: Optional[str] = None
    answer: Optional[str] = None
    faithful: Optional[bool] = None
    judge_rationale: Optional[str] = None
    judge_error: Optional[str] = None


@dataclass
class EvalSummary:
    total: int
    correct: int
    hit_rate: float
    avg_latency_ms: float
    top_k: int
    results: list[QuestionResult] = field(default_factory=list)
    judged: int = 0
    faithful_count: int = 0
    faithfulness_rate: Optional[float] = None


# ── Load questions ─────────────────────────────────────────────────────────────

def load_questions(path: Path) -> list[Question]:
    with open(path) as f:
        raw = json.load(f)
    questions = []
    for item in raw:
        questions.append(Question(
            id=item["id"],
            question=item["question"],
            expected_answer=item.get("expected_answer", ""),
            source_page=int(item["source_page"]),
        ))
    return questions


# ══════════════════════════════════════════════════════════════════════════════
#  MODE 1: API  —  call the running FastAPI server
# ══════════════════════════════════════════════════════════════════════════════

def _api_login(client, base_url: str, email: str, password: str) -> str:
    """Authenticate and return a JWT Bearer token."""
    # pyrefly: ignore [missing-import]
    import httpx
    

    # fastapi-users uses OAuth2 password form
    resp = client.post(
        f"{base_url}/api/auth/jwt/login",
        data={"username": email, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Login failed ({resp.status_code}): {resp.text[:200]}"
        )
    payload = resp.json()
    token = payload.get("access_token") or payload.get("token")
    if not token:
        raise RuntimeError(f"No token in login response: {payload}")
    return token


def _api_create_session(client, base_url: str) -> str:
    """Create a new chat session and return its ID.

    Note: /api/sessions and /api/chat take no document filter — retrieval is scoped to
    all of the authenticated user's ready documents, so --document-id is informational
    only in API mode (it's not sent to the server).
    """
    resp = client.post(f"{base_url}/api/sessions", json={})
    resp.raise_for_status()
    data = resp.json()
    session_id = data.get("id") or data.get("session_id")
    if not session_id:
        raise RuntimeError(f"No session_id in response: {data}")
    return session_id


def _api_extract_pages(data: dict) -> list[int]:
    """
    Pull page numbers out of a /api/chat response.

    Handles a few plausible CitationOut shapes:
      {"page_number": N}  |  {"page": N}  |  {"source_page": N}
    Also handles flat integer lists and nested content arrays.
    """
    pages: list[int] = []

    citations = data.get("assistant_message", {}).get("citations") or data.get("citations") or []
    for citation in citations:
        for key in ("page_number", "page", "source_page"):
            val = citation.get(key)
            if val is not None:
                try:
                    pages.append(int(val))
                except (TypeError, ValueError):
                    pass
                break

    # Fallback: some implementations embed page info in the answer string
    # or return a top-level "pages" list
    if not pages:
        for p in data.get("pages", []):
            try:
                pages.append(int(p))
            except (TypeError, ValueError):
                pass

    return pages


def _api_extract_answer(data: dict) -> str:
    """Pull the assistant's reply text out of a /api/chat response."""
    return data.get("assistant_message", {}).get("content", "") or ""


def _api_extract_chunk_ids(data: dict) -> list[str]:
    citations = data.get("assistant_message", {}).get("citations") or []
    return [c["chunk_id"] for c in citations if c.get("chunk_id")]


def fetch_chunk_texts(chunk_ids: list[str]) -> dict[str, str]:
    """Fetch chunk text for a list of chunk_id strings via a direct DB read.

    Used to reconstruct the actual retrieved context for the faithfulness judge —
    /api/chat's citations only carry page/filename metadata, not the chunk text itself.
    Expects to be run from the repo root so `backend` is importable.
    """
    import uuid as _uuid

    backend_root = Path(__file__).resolve().parent.parent / "backend"
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

    from sqlalchemy import select  # type: ignore

    from app.db.session import async_session_maker  # type: ignore
    from app.models.document import DocumentChunk  # type: ignore

    ids = [_uuid.UUID(c) for c in chunk_ids]

    async def _fetch() -> dict[str, str]:
        async with async_session_maker() as db:
            result = await db.execute(select(DocumentChunk).where(DocumentChunk.id.in_(ids)))
            return {str(row.id): row.text for row in result.scalars().all()}

    return asyncio.run(_fetch())


# ── LLM-as-judge faithfulness check ────────────────────────────────────────────

def judge_faithfulness(
    question: str,
    context: str,
    answer: str,
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    model: str = "google/gemini-2.0-flash-exp:free",
) -> tuple[Optional[bool], str]:
    """Ask an LLM whether `answer` only uses information present in `context`.

    Returns (faithful, rationale_or_error). `faithful` is None if the judge call itself
    failed or its response couldn't be parsed (treated as a judging error, not a verdict).
    """
    # pyrefly: ignore [missing-import]
    import httpx

    prompt = f"""Context: {context}

Question: {question}

Answer: {answer}

Is this answer faithful to the context — meaning it only uses information present in the context and doesn't add facts not mentioned there? Reply with FAITHFUL or UNFAITHFUL and one sentence explaining why."""

    try:
        with httpx.Client(base_url=base_url, timeout=60) as client:
            resp = client.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}]},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        return None, f"judge call failed: {exc}"

    if "choices" not in data:
        message = data.get("error", {}).get("message", "unknown error")
        return None, f"judge upstream error: {message}"

    content = data["choices"][0]["message"]["content"].strip()
    upper = content.upper()
    if "UNFAITHFUL" in upper:
        return False, content
    if "FAITHFUL" in upper:
        return True, content
    return None, f"unparseable judge verdict: {content[:200]}"


def run_api_mode(
    questions: list[Question],
    base_url: str,
    email: str,
    password: str,
    top_k: int,
    judge: bool = False,
    judge_api_key: Optional[str] = None,
    judge_base_url: str = "https://openrouter.ai/api/v1",
    judge_model: str = "google/gemini-2.0-flash-exp:free",
) -> list[QuestionResult]:
    """Fire every question at the live FastAPI server and collect results."""
    try:
        # pyrefly: ignore [missing-import]
        import httpx
    except ImportError:
        sys.exit(
            "httpx is required for API mode.  "
            "Install it:  pip install httpx   or   uv add httpx"
        )

    print(_c(f"\n  Connecting to {base_url} …", DIM))

    results: list[QuestionResult] = []

    with httpx.Client(timeout=60) as client:
        # ── Authenticate ──────────────────────────────────────────────────────
        token = _api_login(client, base_url, email, password)
        client.headers.update({"Authorization": f"Bearer {token}"})
        print(_c("  ✓ Authenticated\n", GREEN))

        for q in questions:
            # Create a fresh session per question so history doesn't leak
            try:
                session_id = _api_create_session(client, base_url)
            except Exception as exc:
                results.append(QuestionResult(
                    id=q.id,
                    question=q.question,
                    source_page=q.source_page,
                    retrieved_pages=[],
                    hit=False,
                    latency_ms=0,
                    error=f"Session creation failed: {exc}",
                ))
                continue

            body = {
                "session_id": session_id,
                "content": q.question,
            }

            t0 = time.perf_counter()
            try:
                resp = client.post(f"{base_url}/api/chat", json=body)
                resp.raise_for_status()
                data = resp.json()
                latency_ms = (time.perf_counter() - t0) * 1000

                pages = _api_extract_pages(data)[:top_k]
                hit = q.source_page in pages
                answer = _api_extract_answer(data)

                faithful: Optional[bool] = None
                rationale = None
                judge_err = None
                if judge:
                    if not judge_api_key:
                        judge_err = "no judge API key configured"
                    else:
                        try:
                            chunk_ids = _api_extract_chunk_ids(data)
                            texts_by_id = fetch_chunk_texts(chunk_ids) if chunk_ids else {}
                            context = "\n\n".join(texts_by_id.get(cid, "") for cid in chunk_ids).strip()
                        except Exception as exc:
                            context = ""
                            judge_err = f"context fetch failed: {exc}"

                        if not judge_err:
                            faithful, rationale_or_err = judge_faithfulness(
                                q.question, context, answer,
                                api_key=judge_api_key, base_url=judge_base_url, model=judge_model,
                            )
                            if faithful is None:
                                judge_err = rationale_or_err
                            else:
                                rationale = rationale_or_err

                results.append(QuestionResult(
                    id=q.id,
                    question=q.question,
                    source_page=q.source_page,
                    retrieved_pages=pages,
                    hit=hit,
                    latency_ms=latency_ms,
                    answer=answer,
                    faithful=faithful,
                    judge_rationale=rationale,
                    judge_error=judge_err,
                ))

            except Exception as exc:
                latency_ms = (time.perf_counter() - t0) * 1000
                results.append(QuestionResult(
                    id=q.id,
                    question=q.question,
                    source_page=q.source_page,
                    retrieved_pages=[],
                    hit=False,
                    latency_ms=latency_ms,
                    error=str(exc),
                ))

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  MODE 2: DIRECT  —  import backend service and call it in-process
# ══════════════════════════════════════════════════════════════════════════════

def run_direct_mode(
    questions: list[Question],
    user_id: str,
    top_k: int,
) -> list[QuestionResult]:
    """
    Import backend.app.services.retrieval directly and run the hybrid
    search → RRF → rerank pipeline without touching the HTTP layer.

    Expects to be invoked from the repo root so that `backend` is on sys.path.
    Note: retrieve_relevant_chunks() is scoped by user_id (not document_id) and
    its candidate count is fixed at TOP_K_FINAL=5 inside the module, so --top-k
    can only narrow the result, not widen it past 5.
    """
    # Add backend to path so absolute imports work
    backend_root = Path(__file__).resolve().parent.parent / "backend"
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

    try:
        import uuid as _uuid

        from app.db.session import async_session_maker      # type: ignore
        from app.services.retrieval import retrieve_relevant_chunks  # type: ignore
    except ImportError as exc:
        sys.exit(
            f"Direct import failed: {exc}\n"
            "Make sure you're running from the repo root and the "
            "virtual environment is activated."
        )

    user_uuid = _uuid.UUID(user_id)

    async def _query(question: str) -> list[int]:
        """Run the retrieval pipeline for one question; return page list."""
        async with async_session_maker() as session:
            chunks = await retrieve_relevant_chunks(question, user_uuid, session)
            return [chunk["page_number"] for chunk in chunks]

    async def _run_all() -> list[QuestionResult]:
        # All questions run inside one event loop — the async engine's pooled
        # asyncpg connections are bound to the loop they were created in, so a
        # fresh asyncio.run() per question (i.e. a fresh loop) breaks the pool
        # with "another operation is in progress" after the first query.
        out: list[QuestionResult] = []
        for q in questions:
            t0 = time.perf_counter()
            try:
                pages = await _query(q.question)
                latency_ms = (time.perf_counter() - t0) * 1000
                hit = q.source_page in pages[:top_k]
                out.append(QuestionResult(
                    id=q.id,
                    question=q.question,
                    source_page=q.source_page,
                    retrieved_pages=pages[:top_k],
                    hit=hit,
                    latency_ms=latency_ms,
                ))
            except Exception as exc:
                latency_ms = (time.perf_counter() - t0) * 1000
                out.append(QuestionResult(
                    id=q.id,
                    question=q.question,
                    source_page=q.source_page,
                    retrieved_pages=[],
                    hit=False,
                    latency_ms=latency_ms,
                    error=str(exc),
                ))
        return out

    return asyncio.run(_run_all())


# ══════════════════════════════════════════════════════════════════════════════
#  Reporting
# ══════════════════════════════════════════════════════════════════════════════

def _truncate(s: str, n: int = 60) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def print_results(results: list[QuestionResult], top_k: int, judge: bool = False) -> EvalSummary:
    """Pretty-print per-question rows and return an EvalSummary."""

    col_w = [4, 62, 6, 18, 6, 10]  # id | question | page | retrieved | hit | ms
    header = (
        f"{'#':>{col_w[0]}}  "
        f"{'Question':<{col_w[1]}}  "
        f"{'Page':>{col_w[2]}}  "
        f"{'Retrieved (top-' + str(top_k) + ')':^{col_w[3]}}  "
        f"{'Hit':^{col_w[4]}}  "
        f"{'ms':>{col_w[5]}}"
    )
    if judge:
        header += f"  {'Faithful':^9}"
    separator = "─" * len(header)

    print()
    print(_c(separator, DIM))
    print(_c(header, BOLD + WHITE))
    print(_c(separator, DIM))

    correct = 0
    total_latency = 0.0
    judged = 0
    faithful_count = 0

    for r in results:
        pages_str = str(r.retrieved_pages) if r.retrieved_pages else "[]"
        hit_str   = "✓" if r.hit else "✗"
        hit_colour = GREEN if r.hit else RED

        if r.error:
            hit_str = "ERR"
            hit_colour = YELLOW

        if r.hit:
            correct += 1

        total_latency += r.latency_ms

        row = (
            f"{r.id:>{col_w[0]}}  "
            f"{_truncate(r.question, col_w[1]):<{col_w[1]}}  "
            f"{r.source_page:>{col_w[2]}}  "
            f"{pages_str:^{col_w[3]}}  "
            f"{hit_str:^{col_w[4]}}  "
            f"{r.latency_ms:>{col_w[5]}.0f}"
        )
        if judge:
            if r.faithful is None:
                faith_str = "err" if r.judge_error else "-"
            else:
                faith_str = "✓" if r.faithful else "✗"
                judged += 1
                if r.faithful:
                    faithful_count += 1
            row += f"  {faith_str:^9}"

        print(_c(row, hit_colour if not r.error else YELLOW))

        if r.error:
            print(_c(f"{'':>{col_w[0]+2}}⚠  {r.error}", DIM + YELLOW))
        elif judge and r.judge_error:
            print(_c(f"{'':>{col_w[0]+2}}⚠  judge: {r.judge_error}", DIM + YELLOW))

    print(_c(separator, DIM))

    total = len(results)
    hit_rate = correct / total if total else 0.0
    avg_latency = total_latency / total if total else 0.0
    faithfulness_rate = faithful_count / judged if judged else None

    return EvalSummary(
        total=total,
        correct=correct,
        hit_rate=hit_rate,
        avg_latency_ms=avg_latency,
        top_k=top_k,
        results=results,
        judged=judged,
        faithful_count=faithful_count,
        faithfulness_rate=faithfulness_rate,
    )


def print_summary(summary: EvalSummary) -> None:
    bar_width = 30
    filled    = round(summary.hit_rate * bar_width)
    bar       = "█" * filled + "░" * (bar_width - filled)
    pct       = summary.hit_rate * 100

    colour = GREEN if pct >= 80 else YELLOW if pct >= 50 else RED

    print()
    print(_c("  ┌─ Evaluation Summary ─────────────────────────┐", BOLD))
    print(_c(f"  │  Hit-Rate @ {summary.top_k:<2}  {bar}  {pct:5.1f}%  │", colour + BOLD))
    print(_c(f"  │  Correct  : {summary.correct} / {summary.total}", DIM) + _c("              │", DIM))
    if summary.faithfulness_rate is not None:
        fpct = summary.faithfulness_rate * 100
        fcolour = GREEN if fpct >= 80 else YELLOW if fpct >= 50 else RED
        print(_c(f"  │  Faithfulness : {fcolour}{fpct:.1f}%{RESET}{DIM} ({summary.faithful_count}/{summary.judged} judged)", DIM) + _c("    │", DIM))
    print(_c(f"  │  Avg latency : {summary.avg_latency_ms:.0f} ms", DIM) + _c("                │", DIM))
    print(_c("  └──────────────────────────────────────────────┘", BOLD))
    print()

    # Machine-readable one-liner for CI / grep
    line = f"HIT_RATE={summary.hit_rate:.4f}  CORRECT={summary.correct}  TOTAL={summary.total}  TOP_K={summary.top_k}"
    if summary.faithfulness_rate is not None:
        line += f"  FAITHFULNESS_RATE={summary.faithfulness_rate:.4f}  FAITHFUL={summary.faithful_count}  JUDGED={summary.judged}"
    print(line)
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DocLens retrieval evaluation — Hit-Rate@K",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--mode",
        choices=["api", "direct"],
        default="api",
        help="'api' calls the running server (default); 'direct' imports backend code in-process.",
    )
    p.add_argument(
        "--questions",
        type=Path,
        default=Path(__file__).parent / "questions.json",
        help="Path to the questions JSON file. Default: eval/questions.json",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks to check. Default: 5",
    )
    p.add_argument(
        "--user-id",
        default=os.getenv("DOCLENS_USER_ID"),
        help="UUID of the user whose documents to query (direct mode only). "
             "Also reads DOCLENS_USER_ID env var.",
    )

    api_group = p.add_argument_group("API mode options")
    api_group.add_argument(
        "--base-url",
        default=os.getenv("DOCLENS_BASE_URL", "http://127.0.0.1:8000"),
        help="DocLens server base URL. Default: http://127.0.0.1:8000",
    )
    api_group.add_argument(
        "--email",
        default=os.getenv("DOCLENS_EMAIL"),
        help="Login e-mail for JWT auth. Also reads DOCLENS_EMAIL.",
    )
    api_group.add_argument(
        "--password",
        default=os.getenv("DOCLENS_PASSWORD"),
        help="Login password. Also reads DOCLENS_PASSWORD.",
    )

    judge_group = p.add_argument_group("Faithfulness judge options")
    judge_group.add_argument(
        "--judge",
        action="store_true",
        help="Also run an LLM-as-judge faithfulness check on each generated answer "
             "against its retrieved context (API mode only).",
    )
    judge_group.add_argument(
        "--judge-api-key",
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key for the judge. Also reads OPENROUTER_API_KEY.",
    )
    judge_group.add_argument(
        "--judge-base-url",
        default=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        help="OpenRouter-compatible base URL for the judge call.",
    )
    judge_group.add_argument(
        "--judge-model",
        default=os.getenv("JUDGE_MODEL", "google/gemini-2.0-flash-exp:free"),
        help="Model the judge uses. Default: google/gemini-2.0-flash-exp:free",
    )

    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write full results as JSON (e.g. eval/results.json).",
    )
    p.add_argument(
        "--fail-under",
        type=float,
        default=None,
        metavar="RATE",
        help="Exit with code 1 if hit_rate is below RATE (e.g. 0.7 for 70%%). "
             "Useful in CI pipelines.",
    )

    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # ── Load questions ────────────────────────────────────────────────────────
    if not args.questions.exists():
        sys.exit(
            f"Questions file not found: {args.questions}\n"
            "Pass --questions <path> or place questions.json in eval/"
        )

    questions = load_questions(args.questions)
    print(
        _c(f"\n  DocLens Eval  ", BOLD + CYAN)
        + _c(
            f"│  {len(questions)} questions  │  top-{args.top_k}  │  mode={args.mode}"
            + ("  │  judge=on" if args.judge else ""),
            DIM,
        )
    )

    # ── Run evaluation ────────────────────────────────────────────────────────
    if args.mode == "api":
        if not args.email or not args.password:
            sys.exit(
                "API mode requires --email and --password "
                "(or env vars DOCLENS_EMAIL / DOCLENS_PASSWORD)."
            )
        if args.judge and not args.judge_api_key:
            sys.exit(
                "--judge requires --judge-api-key (or OPENROUTER_API_KEY env var)."
            )
        results = run_api_mode(
            questions=questions,
            base_url=args.base_url,
            email=args.email,
            password=args.password,
            top_k=args.top_k,
            judge=args.judge,
            judge_api_key=args.judge_api_key,
            judge_base_url=args.judge_base_url,
            judge_model=args.judge_model,
        )
    else:  # direct
        if not args.user_id:
            sys.exit("Direct mode requires --user-id (or DOCLENS_USER_ID env var).")
        results = run_direct_mode(
            questions=questions,
            user_id=args.user_id,
            top_k=args.top_k,
        )

    # ── Display results ───────────────────────────────────────────────────────
    summary = print_results(results, top_k=args.top_k, judge=args.judge)
    print_summary(summary)

    # ── Write JSON output ─────────────────────────────────────────────────────
    if args.json_out:
        out = {
            "hit_rate": summary.hit_rate,
            "correct": summary.correct,
            "total": summary.total,
            "top_k": summary.top_k,
            "avg_latency_ms": round(summary.avg_latency_ms, 1),
            "faithfulness_rate": summary.faithfulness_rate,
            "faithful_count": summary.faithful_count,
            "judged": summary.judged,
            "results": [
                {
                    "id": r.id,
                    "question": r.question,
                    "source_page": r.source_page,
                    "retrieved_pages": r.retrieved_pages,
                    "hit": r.hit,
                    "latency_ms": round(r.latency_ms, 1),
                    **({"error": r.error} if r.error else {}),
                    **({"answer": r.answer} if r.answer else {}),
                    **({"faithful": r.faithful} if r.faithful is not None else {}),
                    **({"judge_rationale": r.judge_rationale} if r.judge_rationale else {}),
                    **({"judge_error": r.judge_error} if r.judge_error else {}),
                }
                for r in results
            ],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(_c(f"  Results written → {args.json_out}", DIM))
        print()

    # ── CI gate ───────────────────────────────────────────────────────────────
    if args.fail_under is not None and summary.hit_rate < args.fail_under:
        sys.exit(
            f"Hit-rate {summary.hit_rate:.2%} is below the required "
            f"threshold {args.fail_under:.2%}. Failing."
        )


if __name__ == "__main__":
    main()