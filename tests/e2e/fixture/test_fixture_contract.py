from __future__ import annotations

from typing import Any

import httpx
import pytest

from fixture_app.app import app


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://fixture.test",
    )


@pytest.mark.asyncio
async def test_reset_creates_isolated_session_state() -> None:
    async with _client() as client:
        reset = await client.post(
            "/__control/reset",
            json={
                "session_id": "session-a",
                "inputs": {
                    "invite_email": "a@example.test",
                    "invite_role": "member",
                    "totp_code": "111222",
                },
            },
        )
        other = await client.get("/__control/state/session-b")

    assert reset.status_code == 200
    assert other.status_code == 200
    assert reset.json()["session_id"] == "session-a"
    assert reset.json()["workspace"]["invited_email"] is None
    assert other.json()["session_id"] == "session-b"
    assert other.json()["fixture_inputs"]["invite_email"] != "a@example.test"


@pytest.mark.asyncio
async def test_invite_completion_contract_matches_both_versions() -> None:
    states: list[dict[str, Any]] = []
    async with _client() as client:
        for version, session_id in (
            ("defective", "invite-defective"),
            ("improved", "invite-improved"),
        ):
            await client.post(
                "/__control/reset",
                json={
                    "session_id": session_id,
                    "inputs": {"invite_email": f"{version}@example.test"},
                },
            )
            page = await client.get(f"/app/{session_id}/{version}")
            assert page.status_code == 200
            if version == "defective":
                assert 'aria-label="Invite teammate"' not in page.text
                assert "Share" in page.text
            else:
                assert 'aria-label="Invite teammate"' in page.text

            result = await client.post(
                f"/app/{session_id}/{version}/team/invite",
                data={
                    "invite_email": f"{version}@example.test",
                    "invite_role": "member",
                },
                follow_redirects=True,
            )
            state = await client.get(f"/__control/state/{session_id}")
            assert result.status_code == 200
            assert "Invitation sent" in result.text
            states.append(state.json())

    assert states[0].keys() == states[1].keys()
    assert states[0]["workspace"].keys() == states[1]["workspace"].keys()
    assert states[0]["security"].keys() == states[1]["security"].keys()
    assert states[0]["completion"]["invite-teammate"] is True
    assert states[1]["completion"]["invite-teammate"] is True


@pytest.mark.asyncio
async def test_two_factor_completion_contract_matches_both_versions() -> None:
    states: list[dict[str, Any]] = []
    async with _client() as client:
        for version, session_id in (
            ("defective", "twofa-defective"),
            ("improved", "twofa-improved"),
        ):
            await client.post(
                "/__control/reset",
                json={"session_id": session_id, "inputs": {"totp_code": "654321"}},
            )
            page = await client.get(f"/app/{session_id}/{version}/settings")
            assert page.status_code == 200
            if version == "defective":
                assert "Two-factor authentication" not in page.text
                assert "Protection" in page.text
            else:
                assert "Security" in page.text
                assert "Two-factor authentication" in page.text

            result = await client.post(
                f"/app/{session_id}/{version}/settings/security/2fa",
                data={"totp_code": "654321"},
                follow_redirects=True,
            )
            state = await client.get(f"/__control/state/{session_id}")
            assert result.status_code == 200
            if version == "defective":
                assert "Protection enabled" in result.text
            else:
                assert "Two-factor authentication enabled" in result.text
            states.append(state.json())

    assert states[0].keys() == states[1].keys()
    assert states[0]["security"].keys() == states[1]["security"].keys()
    assert states[0]["completion"]["enable-2fa"] is True
    assert states[1]["completion"]["enable-2fa"] is True


@pytest.mark.asyncio
async def test_fixture_pages_use_only_same_origin_assets() -> None:
    async with _client() as client:
        response = await client.get("/app/no-outbound/improved")

    assert response.status_code == 200
    assert "http://" not in response.text
    assert "https://" not in response.text
    assert "connect-src 'self'" in response.headers["content-security-policy"]
    assert 'href="/static/app.css"' in response.text
    assert 'src="/static/app.js"' in response.text


@pytest.mark.asyncio
async def test_delete_removes_session_state() -> None:
    async with _client() as client:
        await client.post("/__control/reset", json={"session_id": "to-delete"})
        deleted = await client.delete("/__control/session/to-delete")
        state = await client.get("/__control/state/to-delete")

    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True, "session_id": "to-delete"}
    assert state.status_code == 404
