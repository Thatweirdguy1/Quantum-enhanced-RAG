r"""A minimal demo API, hardened to the pre-deploy checklist.

    uvicorn qrag.serve:app --host 127.0.0.1 --port 8000

Why a research project ships an API at all
------------------------------------------
The retrieval work needs no HTTP surface, and adding one adds attack surface for
nothing -- so this is deliberately small and off by default. It exists because the
supplied "pre-deploy production audit" checklist is written against a service, and
the honest way to satisfy items like *security headers*, *rate limiting*, *CORS not
set to star*, and *no stack traces to the client* is to have a service that
actually implements them, not to write in a report that we would have.

What it does NOT do, on purpose
-------------------------------
No user accounts, no sessions, no cookies, no database. Those checklist items are
recorded as N/A with reasons by ``scripts/security_audit.py`` rather than being
faked. The one authentication mode is a shared bearer token, required only when
``QRAG_ENV=production`` -- and *required* means the process refuses to start
without it, because a service that silently starts open is worse than one that
does not start.

Every response body is generic. Every error carries a correlation id that also
appears in the server log, so an operator can find the traceback that the client
never sees.
"""

from __future__ import annotations

import hmac
import time
import uuid
from collections import deque
from dataclasses import dataclass

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .security import (SECURITY, ResourceLimitExceeded, SecurityError,
                       configure_logging, fingerprint, injection_risk,
                       optional_env, require_env, validate_query, validate_top_k)

LOG = configure_logging()
_ENV = optional_env("QRAG_ENV", "development")
IS_PROD = _ENV.lower() in {"prod", "production"}

# Fail-fast, not fail-open: in production the token is required with no default.
# A missing secret must stop the boot, because the alternative is an unauthenticated
# service that looks healthy.
API_TOKEN = (require_env("QRAG_API_TOKEN",
                         hint="shared bearer token; required when QRAG_ENV=production")
             if IS_PROD else optional_env("QRAG_API_TOKEN", ""))

# CORS: an explicit allowlist. "*" with credentials is the combination the
# checklist singles out, and it is not reachable from this configuration.
ALLOWED_ORIGINS = [o.strip() for o in
                   optional_env("QRAG_ALLOWED_ORIGINS",
                                "http://localhost:5173,http://127.0.0.1:5173").split(",")
                   if o.strip() and o.strip() != "*"]

RATE_LIMIT_PER_MIN = int(optional_env("QRAG_RATE_LIMIT_PER_MIN", "20"))
MAX_BODY_BYTES = 16 * 1024

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    # No inline script and no third-party origins: this API serves JSON only, so
    # the policy can be maximally restrictive without breaking anything.
    "Content-Security-Policy": ("default-src 'none'; frame-ancestors 'none'; "
                                "base-uri 'none'; form-action 'none'"),
}
# HSTS only over TLS. Sending it on plain HTTP in development would pin
# localhost to https in the developer's browser and be a nuisance to undo.
HSTS = "max-age=63072000; includeSubDomains"


@dataclass
class _Bucket:
    """Fixed-window per-IP counter.

    In-process and therefore per-worker: with N workers the effective limit is
    N x RATE_LIMIT_PER_MIN. Stated rather than glossed over -- a real deployment
    needs a shared store (Redis) or a limit at the reverse proxy. For a single-node
    demo this is the honest scope.
    """

    per_min: int
    window: dict = None

    def __post_init__(self) -> None:
        self.window = {}

    def allow(self, key: str, now: float) -> tuple[bool, int]:
        hits = self.window.setdefault(key, deque())
        cutoff = now - 60.0
        while hits and hits[0] < cutoff:
            hits.popleft()
        if len(hits) >= self.per_min:
            return False, int(60 - (now - hits[0])) + 1
        hits.append(now)
        if len(self.window) > 10_000:  # bound the dict itself
            for stale in [k for k, v in self.window.items() if not v][:5_000]:
                self.window.pop(stale, None)
        return True, 0


# The request/response models live at module scope, and so do the fastapi imports
# above, because this module uses ``from __future__ import annotations``: every
# annotation is a string, and fastapi resolves them with typing.get_type_hints,
# which searches module globals only. Defining SearchRequest inside create_app
# made it unresolvable, so fastapi fell back to treating the body parameter as a
# *query* parameter and every POST /search returned 422 -- including valid ones.
# tests/test_qrag.py caught that; the module-level definition is what fixes it.
class SearchRequest(BaseModel):
    query: str = Field(..., max_length=SECURITY.max_query_chars)
    top_k: int = Field(10, ge=1, le=SECURITY.max_top_k)


class Hit(BaseModel):
    doc_id: str
    score: float
    title: str = ""
    injection_risk: str = "none"


class SearchResponse(BaseModel):
    hits: list[Hit]
    n_flagged: int = 0
    latency_ms: float = 0.0
    correlation_id: str = ""


def create_app(pipeline=None, dataset=None) -> FastAPI:
    """Build the FastAPI app. ``pipeline`` is injected so tests need no corpus."""
    app = FastAPI(
        title="Q-RAG demo API",
        version="0.1.0",
        # No interactive docs in production: the checklist lists an exposed
        # Swagger UI as internal-exposure surface.
        docs_url=None if IS_PROD else "/docs",
        redoc_url=None,
        openapi_url=None if IS_PROD else "/openapi.json",
    )
    app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
                       allow_credentials=False, allow_methods=["POST", "GET"],
                       allow_headers=["Authorization", "Content-Type"])
    buckets = _Bucket(RATE_LIMIT_PER_MIN)

    def require_token(authorization: str = Header(default="")) -> None:
        """Bearer check. Constant-time compare, and only enforced in production."""
        if not IS_PROD and not API_TOKEN:
            return
        supplied = authorization[7:] if authorization.startswith("Bearer ") else ""
        if not hmac.compare_digest(supplied, API_TOKEN):
            raise HTTPException(status_code=401, detail="unauthorised")

    @app.middleware("http")
    async def harden(request: Request, call_next):
        correlation = uuid.uuid4().hex[:12]
        request.state.correlation = correlation
        client = request.client.host if request.client else "unknown"

        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            return _error(413, "request body too large", correlation)

        ok, retry = buckets.allow(client, time.monotonic())
        if not ok:
            # Log the fingerprint of the client, not the address: the address is
            # personal data under the data-flow item and is not needed to
            # diagnose a rate limit.
            LOG.warning("rate limit hit client=%s corr=%s",
                        fingerprint(client, 8), correlation)
            resp = _error(429, "rate limit exceeded", correlation)
            resp.headers["Retry-After"] = str(retry)
            return _finish(resp, request)

        try:
            response = await call_next(request)
        except SecurityError as exc:
            # Refusals are the caller's fault and safe to name: they carry a limit,
            # never a path or an internal.
            LOG.info("refused corr=%s: %s", correlation, exc)
            status = 413 if isinstance(exc, ResourceLimitExceeded) else 400
            return _finish(_error(status, str(exc), correlation), request)
        except Exception:
            # The traceback goes to the log with the correlation id; the client
            # gets the id and nothing else. That pairing is the whole point of
            # the checklist's "no stack traces to the client" item.
            LOG.exception("unhandled error corr=%s", correlation)
            return _finish(_error(500, "internal error", correlation), request)
        return _finish(response, request)

    def _error(status: int, message: str, correlation: str) -> JSONResponse:
        return JSONResponse(status_code=status,
                            content={"error": message, "correlation_id": correlation})

    @app.exception_handler(HTTPException)
    async def _http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        """One error envelope for the whole API.

        Without this, fastapi's default handler answers HTTPException with
        ``{"detail": ...}`` while the middleware answers everything else with
        ``{"error": ..., "correlation_id": ...}``. Two shapes means a client has to
        branch on status to find the message, and the 401 from require_token would
        carry no correlation id -- so a failed auth attempt could not be tied to a
        log line. The detail strings are all written here, so surfacing them is safe.
        """
        correlation = getattr(request.state, "correlation", "")
        resp = _error(exc.status_code, str(exc.detail), correlation)
        for header, value in (getattr(exc, "headers", None) or {}).items():
            resp.headers[header] = value
        return _finish(resp, request)

    def _finish(response, request):
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        if request.url.scheme == "https" or IS_PROD:
            response.headers.setdefault("Strict-Transport-Security", HSTS)
        response.headers.setdefault("X-Correlation-Id",
                                    getattr(request.state, "correlation", ""))
        return response

    @app.get("/healthz")
    async def healthz():
        """Liveness only. No version, no config, no dependency status.

        The checklist calls out health endpoints as internal-exposure surface, and
        a readiness probe that reports which model is loaded is reconnaissance.
        """
        return {"status": "ok"}

    @app.post("/search", response_model=SearchResponse)
    async def search(body: SearchRequest, request: Request,
                     _auth: None = Depends(require_token)) -> SearchResponse:
        correlation = request.state.correlation
        t0 = time.perf_counter()
        # Validate again here even though pydantic bounded the field: pydantic
        # checks the length, this normalises Unicode and strips the smuggling
        # channel, and the pipeline is entitled to assume it has been run.
        query = validate_query(body.query)
        top_k = validate_top_k(body.top_k)
        # Log the digest, never the query.
        LOG.info("search corr=%s q=%s k=%d", correlation, fingerprint(query), top_k)

        if pipeline is None:
            raise HTTPException(status_code=503,
                                detail="retrieval index not loaded")

        result = pipeline.retrieve(f"api-{correlation}", query, top_k=top_k)
        hits, flagged = [], 0
        for doc_id in result.ranked[:top_k]:
            title, risk = "", "none"
            if dataset is not None:
                try:
                    doc = dataset.doc(doc_id)
                    title = doc.title[:200]
                    risk = injection_risk(doc.content)
                except KeyError:
                    pass
            flagged += risk != "none"
            hits.append(Hit(doc_id=doc_id, score=result.scores.get(doc_id, 0.0),
                            title=title, injection_risk=risk))
        return SearchResponse(hits=hits, n_flagged=flagged,
                              latency_ms=(time.perf_counter() - t0) * 1e3,
                              correlation_id=correlation)

    return app


app = create_app()


if __name__ == "__main__":
    print(f"env={_ENV} prod={IS_PROD} origins={ALLOWED_ORIGINS} "
          f"rate={RATE_LIMIT_PER_MIN}/min auth={'on' if API_TOKEN else 'off'}")
    print("headers:", ", ".join(SECURITY_HEADERS))
