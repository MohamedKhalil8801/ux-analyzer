from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from ux_analyzer.config.loader import ProjectConfigError, load_project
from ux_analyzer.domain.benchmark import normalize_crawl_url as benchmark_normalize
from ux_analyzer.domain.benchmark import same_origin as benchmark_same_origin
from ux_analyzer.domain.exploration import normalize_crawl_url, same_origin

FIXTURE_PATH = Path(__file__).parents[2] / "fixtures" / "config" / "minimal-project.yaml"
DEMO_PATH = Path(__file__).parents[3] / "benchmarks" / "demo" / "project.yaml"


def _read_project() -> dict[str, Any]:
    with FIXTURE_PATH.open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    assert isinstance(loaded, dict)
    return loaded


def _write_project(tmp_path: Path, project: dict[str, Any], name: str = "project.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    return path


def _base_project() -> dict[str, Any]:
    return _read_project()


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def test_exploration_accepts_multiple_starts_depth_zero(tmp_path: Path) -> None:
    proj = _base_project()
    proj["exploration"] = {
        "start_urls": ["https://a.test/", "https://a.test/about"],
        "depth": 0,
        "max_pages": 2,
    }
    loaded = load_project(_write_project(tmp_path, proj))
    assert loaded.exploration is not None
    assert loaded.exploration.max_pages == 2
    assert len(loaded.exploration.start_urls) == 2
    # also via model normalization
    # depth 0 with sufficient max_pages should be allowed
    # check that start_urls are normalized
    assert "https://a.test/" in loaded.exploration.start_urls
    # second start normalized retains path
    assert any(u.endswith("/about") for u in loaded.exploration.start_urls)


def test_exploration_depth_bounds(tmp_path: Path) -> None:
    for bad_depth in [-1, 6, 10]:
        proj = _base_project()
        proj["exploration"] = {"start_urls": ["https://a.test/"], "depth": bad_depth, "max_pages": 10}
        with pytest.raises(ProjectConfigError):
            load_project(_write_project(tmp_path, proj, f"bad-{bad_depth}.yaml"))
    # edges should pass
    for good in [0, 2, 5]:
        proj = _base_project()
        # for depth 0 need max_pages >= starts (1)
        proj["exploration"] = {"start_urls": ["https://a.test/"], "depth": good, "max_pages": 10}
        loaded = load_project(_write_project(tmp_path, proj, f"good-{good}.yaml"))
        assert loaded.exploration is not None
        assert loaded.exploration.depth == good


def test_exploration_max_pages_bounds(tmp_path: Path) -> None:
    for bad in [0, 201, 500]:
        proj = _base_project()
        proj["exploration"] = {"start_urls": ["https://a.test/"], "depth": 1, "max_pages": bad}
        with pytest.raises(ProjectConfigError):
            load_project(_write_project(tmp_path, proj, f"bad-{bad}.yaml"))
    for good in [1, 50, 200]:
        proj = _base_project()
        proj["exploration"] = {"start_urls": ["https://a.test/"], "depth": 1, "max_pages": good}
        loaded = load_project(_write_project(tmp_path, proj, f"good-{good}.yaml"))
        assert loaded.exploration.max_pages == good


def test_exploration_max_scenarios_bound(tmp_path: Path) -> None:
    for bad in [0, 21, 30]:
        proj = _base_project()
        proj["exploration"] = {
            "start_urls": ["https://a.test/"],
            "depth": 1,
            "max_pages": 10,
            "max_scenarios": bad,
        }
        with pytest.raises(ProjectConfigError):
            load_project(_write_project(tmp_path, proj, f"bad-{bad}.yaml"))
    for good in [1, 8, 20]:
        proj = _base_project()
        proj["exploration"] = {
            "start_urls": ["https://a.test/"],
            "depth": 1,
            "max_pages": 10,
            "max_scenarios": good,
        }
        loaded = load_project(_write_project(tmp_path, proj, f"good-{good}.yaml"))
        assert loaded.exploration.max_scenarios == good


def test_exploration_settle_ms_bounds(tmp_path: Path) -> None:
    for bad in [-1, 15001, 20000]:
        proj = _base_project()
        proj["exploration"] = {
            "start_urls": ["https://a.test/"],
            "depth": 1,
            "max_pages": 10,
            "settle_ms": bad,
        }
        with pytest.raises(ProjectConfigError):
            load_project(_write_project(tmp_path, proj, f"bad-{bad}.yaml"))
    for good in [0, 10000, 15000]:
        proj = _base_project()
        proj["exploration"] = {
            "start_urls": ["https://a.test/"],
            "depth": 1,
            "max_pages": 10,
            "settle_ms": good,
        }
        loaded = load_project(_write_project(tmp_path, proj, f"good-{good}.yaml"))
        assert loaded.exploration.settle_ms == good
    # alias page_settle_ms
    proj = _base_project()
    proj["exploration"] = {
        "start_urls": ["https://a.test/"],
        "depth": 1,
        "max_pages": 10,
        "page_settle_ms": 5000,
    }
    loaded = load_project(_write_project(tmp_path, proj, "alias.yaml"))
    assert loaded.exploration.settle_ms == 5000


def test_exploration_respect_robots_default(tmp_path: Path) -> None:
    proj = _base_project()
    proj["exploration"] = {"start_urls": ["https://a.test/"], "depth": 1, "max_pages": 10}
    loaded = load_project(_write_project(tmp_path, proj))
    assert loaded.exploration.respect_robots is False
    proj["exploration"]["respect_robots"] = True
    loaded2 = load_project(_write_project(tmp_path, proj, "true.yaml"))
    assert loaded2.exploration.respect_robots is True


def test_exploration_duplicate_normalized_starts_rejected(tmp_path: Path) -> None:
    # case-insensitive host duplicate
    proj = _base_project()
    proj["exploration"] = {"start_urls": ["https://example.test/", "https://EXAMPLE.test/"], "depth": 1, "max_pages": 10}
    with pytest.raises(ProjectConfigError, match="unique"):
        load_project(_write_project(tmp_path, proj))
    # query sorted duplicate
    proj = _base_project()
    proj["exploration"] = {
        "start_urls": ["https://example.test/?b=2&a=1", "https://example.test/?a=1&b=2"],
        "depth": 1,
        "max_pages": 10,
    }
    with pytest.raises(ProjectConfigError, match="unique"):
        load_project(_write_project(tmp_path, proj, "q.yaml"))
    # fragment duplicate (fragment stripped)
    proj = _base_project()
    proj["exploration"] = {
        "start_urls": ["https://a.test/page#frag", "https://a.test/page"],
        "depth": 1,
        "max_pages": 10,
    }
    with pytest.raises(ProjectConfigError, match="unique"):
        load_project(_write_project(tmp_path, proj, "frag.yaml"))
    # tracking param duplicate after filtering
    proj = _base_project()
    proj["exploration"] = {
        "start_urls": [
            "https://a.test/p?utm_source=x&a=1&fbclid=123",
            "https://a.test/p?a=1",
        ],
        "depth": 1,
        "max_pages": 10,
    }
    with pytest.raises(ProjectConfigError, match="unique"):
        load_project(_write_project(tmp_path, proj, "track.yaml"))
    # default port duplicate
    proj = _base_project()
    proj["exploration"] = {
        "start_urls": ["https://example.test:443/", "https://example.test/"],
        "depth": 1,
        "max_pages": 10,
    }
    with pytest.raises(ProjectConfigError, match="unique"):
        load_project(_write_project(tmp_path, proj, "port.yaml"))


def test_exploration_query_sort_normalization(tmp_path: Path) -> None:
    # direct helper
    assert benchmark_normalize("https://example.test/p?b=2&a=1") == "https://example.test/p?a=1&b=2"
    assert normalize_crawl_url("https://example.test/p?b=2&a=1") == "https://example.test/p?a=1&b=2"
    # via config loading
    proj = _base_project()
    proj["exploration"] = {
        "start_urls": ["https://example.test/p?b=2&a=1"],
        "depth": 1,
        "max_pages": 10,
    }
    loaded = load_project(_write_project(tmp_path, proj))
    assert loaded.exploration.start_urls[0] == "https://example.test/p?a=1&b=2"


def test_exploration_fragment_stripped(tmp_path: Path) -> None:
    assert benchmark_normalize("https://example.test/page#section") == "https://example.test/page"
    assert normalize_crawl_url("https://example.test/page#section") == "https://example.test/page"
    proj = _base_project()
    proj["exploration"] = {
        "start_urls": ["https://a.test/page#section"],
        "depth": 1,
        "max_pages": 10,
    }
    loaded = load_project(_write_project(tmp_path, proj))
    assert loaded.exploration.start_urls[0] == "https://a.test/page"
    assert "#" not in loaded.exploration.start_urls[0]


def test_exploration_tracking_param_filter(tmp_path: Path) -> None:
    # utm_* and fbclid/gclid removed
    assert benchmark_normalize("https://example.test/p?utm_source=x&a=1&fbclid=123&gclid=abc") == "https://example.test/p?a=1"
    assert normalize_crawl_url("https://example.test/p?utm_source=x&a=1&fbclid=123&gclid=abc") == "https://example.test/p?a=1"
    # combined with sorting and fragment
    assert benchmark_normalize("https://example.test/p?b=2&utm_medium=email&a=1#frag") == "https://example.test/p?a=1&b=2"
    # via config
    proj = _base_project()
    proj["exploration"] = {
        "start_urls": ["https://example.test/p?utm_source=x&a=1&fbclid=123"],
        "depth": 1,
        "max_pages": 10,
    }
    loaded = load_project(_write_project(tmp_path, proj))
    assert loaded.exploration.start_urls[0] == "https://example.test/p?a=1"
    # gclid, gbraid etc
    assert benchmark_normalize("https://a.test/?gclid=1&a=2") == "https://a.test/?a=2"
    assert benchmark_normalize("https://a.test/?utm_campaign=test&x=1") == "https://a.test/?x=1"


def test_exploration_cross_origin_dedup_not_config_error(tmp_path: Path) -> None:
    # Different origins should be allowed, not considered duplicate
    proj = _base_project()
    proj["exploration"] = {
        "start_urls": ["https://a.test/", "https://b.test/"],
        "depth": 1,
        "max_pages": 10,
    }
    loaded = load_project(_write_project(tmp_path, proj))
    assert len(loaded.exploration.start_urls) == 2
    # same_origin should be False
    assert benchmark_same_origin("https://a.test/page", "https://b.test/page") is False
    assert same_origin("https://a.test/page", "https://b.test/page") is False
    # same origin true case
    assert benchmark_same_origin("https://example.test/a", "https://example.test/b") is True
    assert same_origin("https://example.test:443/a", "https://example.test/b") is True


def test_normalize_crawl_url_helpers_via_benchmark_and_exploration() -> None:
    # benchmark supports http default port removal, lowercase, collapse, etc.
    assert benchmark_normalize("https://EXAMPLE.test/") == "https://example.test/"
    assert benchmark_normalize("https://example.test:443/") == "https://example.test/"
    assert benchmark_normalize("http://example.test:80/") == "http://example.test/"
    assert benchmark_normalize("http://example.test:8080/") == "http://example.test:8080/"
    assert benchmark_normalize("https://example.test//a///b") == "https://example.test/a/b"
    # exploration enforces HTTPS
    with pytest.raises(ValueError, match="HTTPS"):
        normalize_crawl_url("http://example.test/")
    # same_origin covers default ports and case
    assert benchmark_same_origin("https://EXAMPLE.test/a", "https://example.test/b") is True
    assert benchmark_same_origin("https://example.test:8443/a", "https://example.test/b") is False


def test_normalize_crawl_url_rejects_userinfo_credentials() -> None:
    bad_urls = [
        "https://user@example.test/",
        "https://user:pass@example.test/",
        "https://:pass@example.test/",
        "https://@example.test/",
    ]
    for bad in bad_urls:
        for fn in (benchmark_normalize, normalize_crawl_url):
            with pytest.raises(ValueError, match="credentials"):
                fn(bad)


def test_normalize_crawl_url_rejects_malformed_ports() -> None:
    bad_ports = [
        "https://example.test:/",  # trailing colon
        "https://example.test:0/",  # port zero
        "https://example.test:port/",  # non-numeric
        "https://example.test:-1/",  # negative
        "https://example.test:99999/",  # out of range
    ]
    for bad in bad_ports:
        for fn in (benchmark_normalize, normalize_crawl_url):
            with pytest.raises(ValueError):
                fn(bad)


def test_normalize_crawl_url_rejects_unicode_and_whitespace_hosts() -> None:
    bad_hosts = [
        "https://exämple.test/",
        "https://例え.test/",
        "https://exämple.test:8443/",
        "https://ex ample.test/",  # internal whitespace
        " https://example.test/",  # leading whitespace (whole URL)
    ]
    for bad in bad_hosts:
        for fn in (benchmark_normalize, normalize_crawl_url):
            with pytest.raises(ValueError, match="host|URL"):
                fn(bad)


def test_normalize_and_same_origin_handle_ipv6_literals() -> None:
    assert benchmark_normalize("https://[2001:DB8::1]/") == "https://[2001:db8::1]/"
    assert normalize_crawl_url("https://[2001:db8::1]/") == "https://[2001:db8::1]/"
    assert (
        normalize_crawl_url("https://[::1]:8443/x//y")
        == "https://[::1]:8443/x/y"
    )
    assert same_origin("https://[2001:DB8::1]/a", "https://[2001:db8::1]/b") is True
    assert same_origin("https://[::1]/a", "https://[::1]:443/b") is True
    assert same_origin("https://[::1]/a", "https://[::2]/b") is False
    assert benchmark_same_origin("https://[::1]:443/a", "https://[::1]/b") is True


def test_same_origin_rejects_hostile_inputs() -> None:
    with pytest.raises(ValueError, match="valid host"):
        same_origin("https://exämple.test/a", "https://example.test/b")
    with pytest.raises(ValueError):
        same_origin("https://example.test:/a", "https://example.test/b")
    with pytest.raises(ValueError):
        same_origin("https://example.test:0/a", "https://example.test/b")
    with pytest.raises(ValueError, match="non-empty"):
        same_origin("   ", "https://example.test/b")


def test_exploration_depth_zero_requires_max_pages_coverage(tmp_path: Path) -> None:
    proj = _base_project()
    proj["exploration"] = {
        "start_urls": ["https://a.test/", "https://a.test/about"],
        "depth": 0,
        "max_pages": 1,
    }
    with pytest.raises(ProjectConfigError, match="max_pages"):
        load_project(_write_project(tmp_path, proj))


def test_exploration_rejects_http_start_url(tmp_path: Path) -> None:
    proj = _base_project()
    proj["exploration"] = {"start_urls": ["http://a.test/"], "depth": 1, "max_pages": 10}
    with pytest.raises(ProjectConfigError):
        load_project(_write_project(tmp_path, proj))


def test_existing_project_without_exploration_still_loads() -> None:
    loaded = load_project(DEMO_PATH)
    assert loaded.exploration is None
    assert loaded.project.id == "attention-guided-demo"
    # also minimal
    loaded2 = load_project(FIXTURE_PATH)
    assert loaded2.exploration is None
