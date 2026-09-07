import asyncio

import pytest

import ux_analyzer.application.run_agent as run_agent_module
from ux_analyzer.application.run_agent import (
    _progress_remaining,
    _ProgressHeartbeat,
    _RunStalled,
    _wait_for_run_progress,
    _WallDeadlineExceeded,
)
from ux_analyzer.domain.benchmark import Budget


def _loop() -> asyncio.AbstractEventLoop:
    return asyncio.get_running_loop()


async def _sleep_beat(heartbeat: _ProgressHeartbeat, seconds: float) -> None:
    await asyncio.sleep(seconds)
    heartbeat.beat()


async def _progress_loop(heartbeat: _ProgressHeartbeat) -> str:
    for _ in range(6):
        await asyncio.sleep(0.01)
        heartbeat.beat()
    return "progressed"


async def test_heartbeat_tracks_idle_time() -> None:
    heartbeat = _ProgressHeartbeat(loop=_loop(), stall_seconds=1.0)
    assert heartbeat.idle_seconds() < 0.2
    await asyncio.sleep(0.05)
    assert 0.04 < heartbeat.idle_seconds() < 0.5
    heartbeat.beat()
    assert heartbeat.idle_seconds() < 0.2


async def test_heartbeat_stall_remaining_without_budget() -> None:
    heartbeat = _ProgressHeartbeat(loop=_loop(), stall_seconds=None)
    assert heartbeat.stall_remaining() is None


async def test_budget_rejects_nonpositive_stall() -> None:
    with pytest.raises(ValueError, match="stall_timeout_seconds"):
        Budget(
            max_steps=1,
            max_observations=1,
            max_interactions=1,
            stall_timeout_seconds=0,
        )


async def test_budget_rejects_stall_above_wall_clock() -> None:
    with pytest.raises(ValueError, match="stall_timeout_seconds"):
        Budget(
            max_steps=1,
            max_observations=1,
            max_interactions=1,
            timeout_seconds=30.0,
            stall_timeout_seconds=60.0,
        )


async def test_budget_accepts_stall_without_wall_clock() -> None:
    budget = Budget(
        max_steps=1,
        max_observations=1,
        max_interactions=1,
        timeout_seconds=None,
        stall_timeout_seconds=90.0,
    )
    assert budget.timeout_seconds is None
    assert budget.stall_timeout_seconds == 90.0


async def test_wait_returns_task_result_without_caps() -> None:
    loop = _loop()
    heartbeat = _ProgressHeartbeat(loop=loop, stall_seconds=None)
    task = asyncio.ensure_future(asyncio.sleep(0.01))
    execution = await _wait_for_run_progress(
        task,
        heartbeat=heartbeat,
        wall_deadline=None,
        stall_seconds=None,
        loop=loop,
    )
    assert execution is None


async def test_watchdog_returns_completed_task() -> None:
    loop = _loop()
    heartbeat = _ProgressHeartbeat(loop=loop, stall_seconds=1.0)
    task = asyncio.ensure_future(asyncio.sleep(0.05, result="done"))
    result = await _wait_for_run_progress(
        task,
        heartbeat=heartbeat,
        wall_deadline=None,
        stall_seconds=1.0,
        loop=loop,
    )
    assert result == "done"


async def test_watchdog_wall_deadline_raises_wall_deadline_sentinel() -> None:
    loop = _loop()
    heartbeat = _ProgressHeartbeat(loop=loop, stall_seconds=None)

    async def hang() -> None:
        await asyncio.sleep(5)

    task = asyncio.ensure_future(hang())
    with pytest.raises(_WallDeadlineExceeded):
        await _wait_for_run_progress(
            task,
            heartbeat=heartbeat,
            wall_deadline=loop.time() + 0.05,
            stall_seconds=None,
            loop=loop,
        )
    assert task.done()


async def test_watchdog_stall_raises_stalled() -> None:
    loop = _loop()
    heartbeat = _ProgressHeartbeat(loop=loop, stall_seconds=0.05)

    async def hang() -> None:
        await asyncio.sleep(5)

    task = asyncio.ensure_future(hang())
    with pytest.raises(_RunStalled, match="stalled: no progress"):
        await _wait_for_run_progress(
            task,
            heartbeat=heartbeat,
            wall_deadline=None,
            stall_seconds=0.05,
            loop=loop,
        )
    assert task.done()


async def test_watchdog_lets_progressing_task_finish() -> None:
    """Beats extend the run past the initial stall window indefinitely."""

    loop = _loop()
    heartbeat = _ProgressHeartbeat(loop=loop, stall_seconds=0.05)
    task = asyncio.ensure_future(_progress_loop(heartbeat))
    result = await _wait_for_run_progress(
        task,
        heartbeat=heartbeat,
        wall_deadline=None,
        stall_seconds=0.05,
        loop=loop,
    )
    assert result == "progressed"


async def test_progress_remaining_mixed_caps() -> None:
    loop = _loop()
    heartbeat = _ProgressHeartbeat(loop=loop, stall_seconds=1.0)
    wall = loop.time() + 100.0
    remaining = _progress_remaining(
        wall_deadline=wall, heartbeat=heartbeat, loop=loop
    )
    assert remaining is not None
    assert remaining <= 1.0


async def test_progress_remaining_unbounded() -> None:
    loop = _loop()
    heartbeat = _ProgressHeartbeat(loop=loop, stall_seconds=None)
    assert (
        _progress_remaining(
            wall_deadline=None, heartbeat=heartbeat, loop=loop
        )
        is None
    )


async def test_inflight_operation_extends_stall_allowance() -> None:
    heartbeat = _ProgressHeartbeat(loop=_loop(), stall_seconds=0.05)
    assert heartbeat.stall_remaining() is not None
    assert heartbeat.stall_remaining() <= 0.05
    heartbeat.begin_operation()
    try:
        assert heartbeat.effective_stall_seconds() == (
            run_agent_module._INFLIGHT_STALL_SECONDS
        )
        assert heartbeat.stall_remaining() > 100.0
    finally:
        heartbeat.end_operation()
    assert heartbeat.in_flight == 0
    assert heartbeat.effective_stall_seconds() == 0.05


async def test_watchdog_lets_slow_inflight_operation_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow but alive operation must not be killed by the base stall window."""

    monkeypatch.setattr(
        run_agent_module, "_INFLIGHT_STALL_SECONDS", 60.0
    )
    loop = _loop()
    heartbeat = _ProgressHeartbeat(loop=loop, stall_seconds=0.05)

    async def slow_operation() -> str:
        heartbeat.begin_operation()
        try:
            await asyncio.sleep(0.15)
        finally:
            heartbeat.end_operation()
        heartbeat.beat()
        return "slow-progress"

    task = asyncio.ensure_future(slow_operation())
    result = await _wait_for_run_progress(
        task,
        heartbeat=heartbeat,
        wall_deadline=None,
        stall_seconds=0.05,
        loop=loop,
    )
    assert result == "slow-progress"


async def test_watchdog_fires_when_inflight_operation_exceeds_inflight_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_agent_module, "_INFLIGHT_STALL_SECONDS", 0.1
    )
    loop = _loop()
    heartbeat = _ProgressHeartbeat(loop=loop, stall_seconds=0.05)

    async def hang_in_flight() -> None:
        heartbeat.begin_operation()
        await asyncio.sleep(5)

    task = asyncio.ensure_future(hang_in_flight())
    with pytest.raises(_RunStalled, match="stalled: no progress"):
        await _wait_for_run_progress(
            task,
            heartbeat=heartbeat,
            wall_deadline=None,
            stall_seconds=0.05,
            loop=loop,
        )
    assert task.done()
