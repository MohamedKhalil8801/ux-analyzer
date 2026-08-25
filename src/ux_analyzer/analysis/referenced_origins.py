"""Best-effort discovery of external origins referenced by a live page."""

from __future__ import annotations

import asyncio
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx

_FETCH_TIMEOUT_SECONDS = 10.0
_NON_RESOURCE_SCHEMES = frozenset(
    {"", "about", "blob", "data", "javascript", "mailto", "tel"}
)


class _ReferencedOriginParser(HTMLParser):
    """Collect absolute origins from resource-bearing attributes."""

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self._base_url = base_url
        self._origins: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name: value or "" for name, value in attrs}
        candidates: tuple[str, ...] = ()
        if tag in {"script", "img", "iframe", "source", "video", "audio", "embed"}:
            # Load-time subresources: blocking these breaks the page render.
            candidates = (
                attributes.get("src", ""),
                attributes.get("srcset", ""),
                attributes.get("poster", ""),
            )
        elif tag == "link":
            # Stylesheets/preloads load eagerly; plain anchors are navigation
            # targets the policy handles separately at click time.
            candidates = (attributes.get("href", ""), attributes.get("srcset", ""))
        for candidate in candidates:
            for token in candidate.split(","):
                target = token.strip().split(" ")[0]
                if target:
                    self._record(target)

    def _record(self, target: str) -> None:
        resolved = urljoin(self._base_url, target)
        parsed = urlsplit(resolved)
        if parsed.scheme.lower() in _NON_RESOURCE_SCHEMES or not parsed.netloc:
            return
        hostname = parsed.hostname.lower()
        if not hostname:
            return
        try:
            port = parsed.port
        except ValueError:
            return
        default_port = 80 if parsed.scheme.lower() == "http" else 443
        suffix = f":{port}" if port is not None and port != default_port else ""
        self._origins.add(f"{parsed.scheme.lower()}://{hostname}{suffix}")

    @property
    def origins(self) -> frozenset[str]:
        return frozenset(self._origins)


def _origin_of(url: str) -> str:
    parsed = urlsplit(url)
    host = parsed.hostname.lower()
    try:
        port = parsed.port
    except ValueError:
        port = None
    default_port = 80 if parsed.scheme.lower() == "http" else 443
    suffix = f":{port}" if port is not None and port != default_port else ""
    return f"{parsed.scheme.lower()}://{host}{suffix}"


async def referenced_origins(
    url: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> frozenset[str]:
    """Return foreign origins a page references; empty on any failure."""

    owned_client = client is None
    fetched_client = client or httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(_FETCH_TIMEOUT_SECONDS),
    )
    try:
        response = await fetched_client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "html" not in content_type.lower():
            return frozenset()
        final_url = str(response.url)
        parser = _ReferencedOriginParser(final_url)
        parser.feed(response.text)
        return parser.origins - {_origin_of(final_url)}
    except Exception:  # noqa: BLE001 - pre-flight discovery is best-effort
        return frozenset()
    finally:
        if owned_client:
            await fetched_client.aclose()


def referenced_origins_sync(url: str) -> frozenset[str]:
    """Synchronous wrapper safe to call outside a running loop."""

    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None
    if running_loop is not None:
        raise RuntimeError("referenced_origins_sync cannot run inside a loop")
    return asyncio.run(referenced_origins(url))
