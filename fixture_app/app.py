"""Bundled FastAPI SaaS fixture used by benchmark and browser tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from fixture_app.state import SessionState, session_store

Version = Literal["defective", "improved"]
FIXTURE_ROOT = Path(__file__).parent
templates = Jinja2Templates(directory=str(FIXTURE_ROOT / "templates"))

app = FastAPI(title="Controlled SaaS Fixture", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=FIXTURE_ROOT / "static"), name="static")


class ResetRequest(BaseModel):
    """Private reset payload containing fake scenario inputs."""

    session_id: str = Field(min_length=1)
    inputs: dict[str, str] = Field(default_factory=dict)


@app.middleware("http")
async def local_only_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'none'"
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.post("/__control/reset")
async def reset_session(payload: ResetRequest = Body(...)) -> JSONResponse:
    state = session_store.reset(payload.session_id, payload.inputs)
    return JSONResponse(state.as_dict())


@app.get("/__control/state/{session_id}")
async def get_session_state(session_id: str) -> JSONResponse:
    snapshot = session_store.snapshot(session_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="session not found")
    return JSONResponse(snapshot)


@app.delete("/__control/session/{session_id}")
async def delete_session(session_id: str) -> JSONResponse:
    session_store.delete(session_id)
    return JSONResponse({"deleted": True, "session_id": session_id})


@app.get("/app/{session_id}/{version}", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    session_id: str,
    version: str,
    notice: str | None = Query(default=None),
) -> HTMLResponse:
    checked_version = _check_version(version)
    state = session_store.get_or_create(session_id)
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context=_page_context(
            request=request,
            session_id=session_id,
            version=checked_version,
            state=state,
            screen="dashboard",
            notice=notice,
        ),
    )


@app.get("/app/{session_id}/{version}/team", response_class=HTMLResponse)
async def team_page(
    request: Request,
    session_id: str,
    version: str,
    notice: str | None = Query(default=None),
) -> HTMLResponse:
    checked_version = _check_version(version)
    state = session_store.get_or_create(session_id)
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context=_page_context(
            request=request,
            session_id=session_id,
            version=checked_version,
            state=state,
            screen="team",
            notice=notice,
        ),
    )


@app.post("/app/{session_id}/{version}/team/invite")
async def invite_teammate(
    request: Request,
    session_id: str,
    version: str,
) -> RedirectResponse:
    checked_version = _check_version(version)
    values = await _request_values(request)
    state = session_store.get_or_create(session_id)
    email = values.get("invite_email", state.fixture_inputs["invite_email"]).strip()
    role = values.get("invite_role", state.fixture_inputs["invite_role"]).strip()
    if not email or "@" not in email or not role:
        return RedirectResponse(
            _route(checked_version, session_id, "team", "invalid-invite"),
            status_code=303,
        )
    session_store.invite(session_id, email, role)
    return RedirectResponse(
        _route(checked_version, session_id, "team", "invite-sent"),
        status_code=303,
    )


@app.get("/app/{session_id}/{version}/settings", response_class=HTMLResponse)
async def settings(
    request: Request,
    session_id: str,
    version: str,
    notice: str | None = Query(default=None),
    error: str | None = Query(default=None),
) -> HTMLResponse:
    checked_version = _check_version(version)
    state = session_store.get_or_create(session_id)
    return templates.TemplateResponse(
        request=request,
        name="settings.html",
        context=_page_context(
            request=request,
            session_id=session_id,
            version=checked_version,
            state=state,
            screen="settings",
            notice=notice,
            error=error,
        ),
    )


@app.post("/app/{session_id}/{version}/settings/security/2fa")
async def enable_two_factor(
    request: Request,
    session_id: str,
    version: str,
) -> RedirectResponse:
    checked_version = _check_version(version)
    values = await _request_values(request)
    state = session_store.get_or_create(session_id)
    code = values.get("totp_code", "").strip()
    if code != state.fixture_inputs["totp_code"]:
        return RedirectResponse(
            _settings_route(checked_version, session_id, error="invalid-code"),
            status_code=303,
        )
    session_store.enable_two_factor(session_id)
    return RedirectResponse(
        _settings_route(checked_version, session_id, notice="two-factor-enabled"),
        status_code=303,
    )


def _check_version(version: str) -> Version:
    if version not in {"defective", "improved"}:
        raise HTTPException(status_code=404, detail="unknown fixture version")
    return version  # type: ignore[return-value]


def _page_context(
    *,
    request: Request,
    session_id: str,
    version: Version,
    state: SessionState,
    screen: str,
    notice: str | None = None,
    error: str | None = None,
) -> dict[str, object]:
    return {
        "request": request,
        "session_id": session_id,
        "version": version,
        "state": state,
        "screen": screen,
        "notice": notice,
        "error": error,
        "fixture_inputs": state.fixture_inputs,
    }


def _route(version: Version, session_id: str, page: str, notice: str) -> str:
    return f"/app/{session_id}/{version}/{page}?notice={notice}"


def _settings_route(version: Version, session_id: str, **query: str) -> str:
    query_string = "&".join(f"{key}={value}" for key, value in query.items())
    suffix = f"?{query_string}" if query_string else ""
    return f"/app/{session_id}/{version}/settings{suffix}"


async def _request_values(request: Request) -> dict[str, str]:
    body = await request.body()
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError:
            return {}
        if isinstance(decoded, dict):
            return {key: str(value) for key, value in decoded.items()}
        return {}
    parsed = parse_qs(body.decode("utf-8"), keep_blank_values=True)
    return {key: values[0] for key, values in parsed.items() if values}
