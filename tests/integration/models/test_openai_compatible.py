from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from ux_analyzer.adapters import openai as openai_adapter
from ux_analyzer.adapters.openai import (
    CodexStructuredClient,
    ModelFailureError,
    OpenAICompatibleSettings,
    OpenAICompatibleStructuredClient,
    create_structured_model_client,
)
from ux_analyzer.ports.models import ChatMessage, ModelRole
from ux_analyzer.providers.cognitive import CognitiveModelResponse
from ux_analyzer.providers.scent import CoarseScentResponse


def _settings(**overrides: object) -> OpenAICompatibleSettings:
    values: dict[str, object] = {
        "base_url": "https://fake-llm.test/v1",
        "api_key": "secret-api-key",
        "scent_model": "scent-model",
        "cognitive_model": "cognitive-model",
        "retry_policy": {"max_attempts": 3, "base_delay_seconds": 0},
    }
    values.update(overrides)
    return OpenAICompatibleSettings.model_validate(values)


class _FakeCodexProcess:
    def __init__(self, returncode: int, stderr: bytes = b"codex stderr must not be recorded") -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.input: bytes | None = None

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        self.input = input
        return (
            b"codex stdout must not be recorded",
            self.stderr,
        )


class _NeverCompletingCodexProcess:
    pid = 4242

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.communicate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0
        self.started = asyncio.Event()
        self.never_complete = asyncio.Event()

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        del input
        self.communicate_calls += 1
        self.started.set()
        await self.never_complete.wait()
        return b"", b""

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9
        self.never_complete.set()

    async def wait(self) -> int:
        self.wait_calls += 1
        while self.returncode is None:
            await self.never_complete.wait()
        return self.returncode or -9


class _NeverReapingCodexProcess(_NeverCompletingCodexProcess):
    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    async def wait(self) -> int:
        self.wait_calls += 1
        try:
            await self.never_complete.wait()
        except asyncio.CancelledError:
            await self.never_complete.wait()
        return self.returncode or -9


class _FakeTaskkillProcess:
    pid = 5252

    def __init__(self, returncode: int) -> None:
        self.returncode = returncode
        self.kill_calls = 0

    def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9


def _patch_codex_process(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[tuple[object, ...], dict[str, object]]],
    processes: list[_FakeCodexProcess],
    schemas: list[object],
    *,
    output: str | None,
    returncode: int,
    stderr: bytes = b"codex stderr must not be recorded",
) -> None:
    async def create_subprocess_exec(
        *args: object, **kwargs: object
    ) -> _FakeCodexProcess:
        calls.append((args, kwargs))
        schema_path = Path(args[args.index("--output-schema") + 1])
        schemas.append(json.loads(schema_path.read_text(encoding="utf-8")))
        output_path = Path(args[args.index("--output-last-message") + 1])
        if output is not None:
            output_path.write_text(output, encoding="utf-8")
        process = _FakeCodexProcess(returncode, stderr)
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)


def _patch_never_completing_process(
    monkeypatch: pytest.MonkeyPatch,
    process: _NeverCompletingCodexProcess,
) -> None:
    async def create_subprocess_exec(
        *_args: object, **_kwargs: object
    ) -> _NeverCompletingCodexProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)


def _patch_tree_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[bool, float]],
) -> None:
    async def terminate_tree(
        process: object,
        *,
        process_group_id: int | None,
        force: bool,
        deadline: float,
    ) -> bool:
        del process_group_id
        calls.append((force, deadline))
        if force:
            cast_process = process
            cast_process.kill()  # type: ignore[attr-defined]
        return True

    monkeypatch.setattr(openai_adapter, "_terminate_codex_process_tree", terminate_tree)


@pytest.mark.asyncio
async def test_codex_structured_client_runs_read_only_command_and_validates_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    processes: list[_FakeCodexProcess] = []
    schemas: list[object] = []
    _patch_codex_process(
        monkeypatch,
        calls,
        processes,
        schemas,
        output=json.dumps({"scores": [{"element_id": "target", "score": 0.7}]}),
        returncode=0,
    )
    client = CodexStructuredClient(
        _settings(
            mode="codex",
            retry_policy={"max_attempts": 1, "base_delay_seconds": 0},
        )
    )

    result = await client.complete(
        CoarseScentResponse,
        (ChatMessage(role="user", content="Find invite"),),
        model="gpt-scent",
        role=ModelRole.COARSE_SCENT,
    )

    assert result.scores[0].element_id == "target"
    assert result.scores[0].score == pytest.approx(0.7)
    args, kwargs = calls[0]
    assert args[0:8] == (
        "codex",
        "exec",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--model",
        "gpt-scent",
        "--output-schema",
    )
    assert args[-1] == "-"
    assert "--output-last-message" in args
    response_path = Path(args[args.index("--output-last-message") + 1])
    assert schemas == [CoarseScentResponse.model_json_schema()]
    assert response_path.name == "response.json"
    assert kwargs["stdin"] is asyncio.subprocess.PIPE
    assert kwargs["stdout"] is asyncio.subprocess.PIPE
    assert kwargs["stderr"] is asyncio.subprocess.PIPE
    assert json.loads(processes[0].input.decode("utf-8")) == [
        {"role": "user", "content": "Find invite"}
    ]
    assert client.endpoint_origin == "codex-cli"
    manifest = client.manifest(ModelRole.COARSE_SCENT, "gpt-scent")
    assert manifest.provider_id == "codex-cli"
    assert manifest.provider_version == "codex-cli"
    assert manifest.endpoint_origin == "codex-cli"
    assert client.records[0].request["role"] == ModelRole.COARSE_SCENT.value
    assert client.records[0].request["model"] == "gpt-scent"
    assert client.records[0].request["messages"] == [
        {"role": "user", "content": "Find invite"}
    ]
    assert client.records[0].request["schema_version"] == "scent-coarse-v1"
    assert client.records[0].response == {"status": "success"}
    assert "codex stdout" not in json.dumps(client.records[0].request)
    assert "codex stderr" not in json.dumps(client.records[0].response)


@pytest.mark.asyncio
async def test_codex_process_starts_in_terminable_group_or_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    processes: list[_FakeCodexProcess] = []
    schemas: list[object] = []
    _patch_codex_process(
        monkeypatch,
        calls,
        processes,
        schemas,
        output=json.dumps({"scores": []}),
        returncode=0,
    )
    client = CodexStructuredClient(
        _settings(
            mode="codex",
            retry_policy={"max_attempts": 1, "base_delay_seconds": 0},
        )
    )
    await client.complete(
        CoarseScentResponse,
        (ChatMessage(role="user", content="Find invite"),),
        model="gpt-scent",
        role=ModelRole.COARSE_SCENT,
    )

    kwargs = calls[0][1]
    if os.name == "nt":
        assert kwargs["creationflags"] & subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert kwargs["start_new_session"] is True


@pytest.mark.asyncio
async def test_codex_timeout_bounds_descendant_like_cleanup_and_records_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _NeverCompletingCodexProcess()
    _patch_never_completing_process(monkeypatch, process)
    tree_cleanup_calls: list[tuple[bool, float]] = []
    _patch_tree_cleanup(monkeypatch, tree_cleanup_calls)
    monkeypatch.setattr(openai_adapter, "_CODEX_CLEANUP_TIMEOUT_SECONDS", 0.01)
    client = CodexStructuredClient(
        _settings(
            mode="codex",
            timeout_seconds=0.01,
            retry_policy={"max_attempts": 1, "base_delay_seconds": 0},
        )
    )
    def fail_wait_for(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("cleanup must use deadline-aware task waits")

    monkeypatch.setattr(asyncio, "wait_for", fail_wait_for)
    with pytest.raises(ModelFailureError, match="timeout"):
        async with asyncio.timeout(0.2):
            await client.complete(
                CoarseScentResponse,
                (ChatMessage(role="user", content="Find invite"),),
                model="gpt-scent",
                role=ModelRole.COARSE_SCENT,
            )

    expected_forces = [True] if os.name == "nt" else [False, True]
    assert [force for force, _deadline in tree_cleanup_calls] == expected_forces
    assert len({deadline for _force, deadline in tree_cleanup_calls}) == 1
    assert process.communicate_calls >= 1
    assert process.kill_calls >= 1
    assert process.wait_calls >= 1
    assert client.retry_events == ()
    assert client.records[0].attempts == 1
    assert client.records[0].response == {"failure": "timeout"}


@pytest.mark.asyncio
async def test_codex_outer_cancellation_cleans_tree_and_reaps_direct_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _NeverCompletingCodexProcess()
    _patch_never_completing_process(monkeypatch, process)
    tree_cleanup_calls: list[tuple[bool, float]] = []
    _patch_tree_cleanup(monkeypatch, tree_cleanup_calls)
    monkeypatch.setattr(openai_adapter, "_CODEX_CLEANUP_TIMEOUT_SECONDS", 0.01)
    client = CodexStructuredClient(
        _settings(
            mode="codex",
            timeout_seconds=None,
            retry_policy={"max_attempts": 1, "base_delay_seconds": 0},
        )
    )

    task = asyncio.create_task(
        client.complete(
            CoarseScentResponse,
            (ChatMessage(role="user", content="Find invite"),),
            model="gpt-scent",
            role=ModelRole.COARSE_SCENT,
        )
    )
    await process.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(0.2):
            await task

    expected_forces = [True] if os.name == "nt" else [False, True]
    assert [force for force, _deadline in tree_cleanup_calls] == expected_forces
    assert process.kill_calls >= 1
    assert process.wait_calls >= 1
    assert client.records == ()


@pytest.mark.asyncio
async def test_codex_cleanup_failure_records_sanitized_process_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _NeverReapingCodexProcess()
    _patch_never_completing_process(monkeypatch, process)
    tree_cleanup_calls: list[tuple[bool, float]] = []
    _patch_tree_cleanup(monkeypatch, tree_cleanup_calls)
    monkeypatch.setattr(openai_adapter, "_CODEX_CLEANUP_TIMEOUT_SECONDS", 0.02)
    client = CodexStructuredClient(
        _settings(
            mode="codex",
            timeout_seconds=0.01,
            retry_policy={"max_attempts": 1, "base_delay_seconds": 0},
        )
    )

    with pytest.raises(ModelFailureError, match="process-error"):
        async with asyncio.timeout(0.2):
            await client.complete(
                CoarseScentResponse,
                (ChatMessage(role="user", content="Find invite"),),
                model="gpt-scent",
                role=ModelRole.COARSE_SCENT,
            )

    assert client.records[0].response == {"failure": "process-error"}


@pytest.mark.asyncio
async def test_codex_cancellation_before_spawn_handle_recovers_and_cleans_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _NeverCompletingCodexProcess()
    spawn_started = asyncio.Event()
    release_spawn = asyncio.Event()
    tree_cleanup_calls: list[tuple[bool, float]] = []
    _patch_tree_cleanup(monkeypatch, tree_cleanup_calls)
    monkeypatch.setattr(openai_adapter, "_CODEX_CLEANUP_TIMEOUT_SECONDS", 0.05)

    async def delayed_spawn(
        *_args: object, **_kwargs: object
    ) -> _NeverCompletingCodexProcess:
        spawn_started.set()
        await release_spawn.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
    client = CodexStructuredClient(
        _settings(
            mode="codex",
            timeout_seconds=None,
            retry_policy={"max_attempts": 1, "base_delay_seconds": 0},
        )
    )
    task = asyncio.create_task(
        client.complete(
            CoarseScentResponse,
            (ChatMessage(role="user", content="Find invite"),),
            model="gpt-scent",
            role=ModelRole.COARSE_SCENT,
        )
    )
    await spawn_started.wait()
    task.cancel()
    release_spawn.set()

    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(0.2):
            await task

    expected_forces = [True] if os.name == "nt" else [False, True]
    assert [force for force, _deadline in tree_cleanup_calls] == expected_forces
    assert process.kill_calls >= 1
    assert process.wait_calls >= 1


@pytest.mark.asyncio
async def test_posix_cleanup_escalates_group_after_leader_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name == "nt" or not hasattr(os, "killpg"):
        pytest.skip("POSIX process groups unavailable")
    process = _NeverCompletingCodexProcess()
    communication_task = asyncio.create_task(process.communicate())
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(openai_adapter.os, "getpgid", lambda _pid: 7777)

    def record_signal(group_id: int, signal_number: int) -> None:
        signals.append((group_id, signal_number))
        if signal_number == 15:
            process.returncode = -15

    monkeypatch.setattr(openai_adapter.os, "killpg", record_signal)
    result = await openai_adapter._cleanup_codex_process(
        process,
        process_group_id=7777,
        communication_task=communication_task,
        deadline=asyncio.get_running_loop().time() + 0.2,
    )

    assert result.success
    assert signals == [(7777, 15), (7777, 9)]
    assert communication_task.done()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_mode", ("returncode", "unavailable"))
async def test_windows_taskkill_failure_is_surface_as_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    if os.name != "nt":
        pytest.skip("Windows taskkill unavailable")
    process = _NeverCompletingCodexProcess()
    taskkill = _FakeTaskkillProcess(1)

    def create_taskkill(*args: object, **kwargs: object) -> _FakeTaskkillProcess:
        del args, kwargs
        if failure_mode == "unavailable":
            raise FileNotFoundError("taskkill")
        return taskkill

    monkeypatch.setattr(openai_adapter.subprocess, "Popen", create_taskkill)
    result = await openai_adapter._cleanup_codex_process(
        process,
        process_group_id=None,
        communication_task=None,
        deadline=asyncio.get_running_loop().time() + 0.2,
    )

    assert not result.success
    assert process.kill_calls >= 1


@pytest.mark.asyncio
async def test_cleanup_reports_final_reap_failure_by_hard_deadline() -> None:
    process = _NeverReapingCodexProcess()
    deadline = asyncio.get_running_loop().time() + 0.05

    async with asyncio.timeout(0.2):
        result = await openai_adapter._cleanup_codex_process(
            process,
            process_group_id=None,
            communication_task=None,
            deadline=deadline,
        )

    assert not result.success


async def _pid_exists(pid: int) -> bool:
    if os.name == "nt":
        probe = await asyncio.create_subprocess_exec(
            "tasklist",
            "/FI",
            f"PID eq {pid}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        output, _ = await probe.communicate()
        return str(pid).encode() in output
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.mark.asyncio
async def test_real_process_tree_cleanup_kills_parent_and_child() -> None:
    if os.name == "nt" and not hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        pytest.skip("Windows process groups unavailable")
    if os.name != "nt" and not hasattr(os, "killpg"):
        pytest.skip("POSIX process groups unavailable")

    script = (
        "import subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "print(child.pid, flush=True); time.sleep(60)"
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        **openai_adapter._codex_process_options(),
    )
    assert process.stdout is not None
    child_pid = int((await asyncio.wait_for(process.stdout.readline(), 0.5)).strip())
    if os.name == "nt":
        assert child_pid in openai_adapter._windows_descendant_pids(process.pid)
    communication_task = asyncio.create_task(process.communicate())
    process_group_id = openai_adapter._codex_process_group_id(process)

    result = await openai_adapter._cleanup_codex_process(
        process,
        process_group_id=process_group_id,
        communication_task=communication_task,
        deadline=asyncio.get_running_loop().time() + 1.0,
    )

    assert result.success
    for _ in range(20):
        if not await _pid_exists(child_pid):
            break
        await asyncio.sleep(0.01)
    assert not await _pid_exists(child_pid)
    assert process.returncode is not None


@pytest.mark.asyncio
async def test_codex_unbounded_timeout_does_not_wrap_process_communication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    processes: list[_FakeCodexProcess] = []
    schemas: list[object] = []
    _patch_codex_process(
        monkeypatch,
        calls,
        processes,
        schemas,
        output=json.dumps({"scores": [{"element_id": "target", "score": 0.7}]}),
        returncode=0,
    )

    def fail_wait_for(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unbounded Codex call must not use asyncio.wait_for")

    monkeypatch.setattr(asyncio, "wait_for", fail_wait_for)
    client = CodexStructuredClient(
        _settings(
            mode="codex",
            timeout_seconds=None,
            retry_policy={"max_attempts": 1, "base_delay_seconds": 0},
        )
    )

    result = await client.complete(
        CoarseScentResponse,
        (ChatMessage(role="user", content="Find invite"),),
        model="gpt-scent",
        role=ModelRole.COARSE_SCENT,
    )

    assert result.scores[0].score == pytest.approx(0.7)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "schema", "model", "expected_effort", "output"),
    (
        (
            ModelRole.COARSE_SCENT,
            CoarseScentResponse,
            "gpt-scent",
            "max",
            '{"scores":[{"element_id":"target","score":0.7}]}',
        ),
        (
            ModelRole.COGNITIVE,
            CognitiveModelResponse,
            "gpt-cognitive",
            "high",
            '{"action":"abandon","reason":"No supported control is visible."}',
        ),
    ),
)
async def test_codex_forwards_role_reasoning_effort_to_cli(
    monkeypatch: pytest.MonkeyPatch,
    role: ModelRole,
    schema: type[BaseModel],
    model: str,
    expected_effort: str,
    output: str,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    processes: list[_FakeCodexProcess] = []
    schemas: list[object] = []
    _patch_codex_process(
        monkeypatch,
        calls,
        processes,
        schemas,
        output=output,
        returncode=0,
    )
    client = CodexStructuredClient(
        _settings(
            mode="codex",
            scent_reasoning_effort="max",
            cognitive_reasoning_effort="high",
            retry_policy={"max_attempts": 1, "base_delay_seconds": 0},
        )
    )

    await client.complete(
        schema,
        (ChatMessage(role="user", content="Choose next step"),),
        model=model,
        role=role,
    )

    args = calls[0][0]
    config_index = args.index("-c")
    assert args[config_index + 1] == f"model_reasoning_effort={expected_effort}"
    assert client.records[0].request["reasoning_effort"] == expected_effort


@pytest.mark.asyncio
async def test_codex_cognitive_schema_requires_nullable_root_fields_without_changing_pydantic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    processes: list[_FakeCodexProcess] = []
    schemas: list[object] = []
    _patch_codex_process(
        monkeypatch,
        calls,
        processes,
        schemas,
        output=json.dumps(
            {
                "action": "inspect",
                "element_id": "target",
                "reason": "Inspect visible control.",
            }
        ),
        returncode=0,
    )
    client = CodexStructuredClient(
        _settings(
            mode="codex",
            retry_policy={"max_attempts": 1, "base_delay_seconds": 0},
        )
    )

    result = await client.complete(
        CognitiveModelResponse,
        (ChatMessage(role="user", content="Choose an action"),),
        model="gpt-cognitive",
        role=ModelRole.COGNITIVE,
    )

    emitted_schema = schemas[0]
    assert isinstance(emitted_schema, dict)
    properties = emitted_schema["properties"]
    assert isinstance(properties, dict)
    assert emitted_schema["required"] == list(properties)
    assert all(
        any(option.get("type") == "null" for option in property_schema["anyOf"])
        for property_schema in properties.values()
    )
    assert CognitiveModelResponse.model_json_schema().get("required") is None
    assert result.action == "inspect"
    assert result.element_id == "target"


@pytest.mark.asyncio
async def test_codex_process_exit_retries_until_attempt_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    processes: list[_FakeCodexProcess] = []
    schemas: list[object] = []
    _patch_codex_process(
        monkeypatch,
        calls,
        processes,
        schemas,
        output=None,
        returncode=1,
    )
    client = CodexStructuredClient(
        _settings(
            mode="codex",
            retry_policy={"max_attempts": 2, "base_delay_seconds": 0},
        )
    )

    with pytest.raises(ModelFailureError, match="process-exit"):
        await client.complete(
            CoarseScentResponse,
            (ChatMessage(role="user", content="Find invite"),),
            model="gpt-scent",
            role=ModelRole.COARSE_SCENT,
        )

    assert len(calls) == 2
    assert len(client.retry_events) == 1
    assert client.retry_events[0].reason == "process-exit"
    assert client.records[0].attempts == 2
    assert client.records[0].response == {"failure": "process-exit"}


@pytest.mark.asyncio
async def test_codex_rate_limit_process_failure_is_classified_and_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    processes: list[_FakeCodexProcess] = []
    schemas: list[object] = []
    _patch_codex_process(
        monkeypatch,
        calls,
        processes,
        schemas,
        output=None,
        returncode=1,
        stderr=b"429 Too Many Requests: rate limit exceeded",
    )
    client = CodexStructuredClient(
        _settings(
            mode="codex",
            retry_policy={"max_attempts": 1, "base_delay_seconds": 0},
        )
    )

    with pytest.raises(ModelFailureError, match="rate-limit"):
        await client.complete(
            CoarseScentResponse,
            (ChatMessage(role="user", content="Find invite"),),
            model="gpt-scent",
            role=ModelRole.COARSE_SCENT,
        )

    assert client.records[0].response == {"failure": "rate-limit"}


@pytest.mark.asyncio
async def test_codex_invalid_json_retries_with_existing_structured_output_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    processes: list[_FakeCodexProcess] = []
    schemas: list[object] = []
    _patch_codex_process(
        monkeypatch,
        calls,
        processes,
        schemas,
        output="not json",
        returncode=0,
    )
    client = CodexStructuredClient(
        _settings(
            mode="codex",
            retry_policy={"max_attempts": 2, "base_delay_seconds": 0},
        )
    )

    with pytest.raises(ModelFailureError, match="invalid structured output"):
        await client.complete(
            CoarseScentResponse,
            (ChatMessage(role="user", content="Find invite"),),
            model="gpt-scent",
            role=ModelRole.COARSE_SCENT,
        )

    assert len(calls) == 2
    assert len(client.retry_events) == 1
    assert client.retry_events[0].reason == "invalid-structured-output"
    assert client.records[0].response == {"failure": "invalid-structured-output"}


@pytest.mark.asyncio
async def test_structured_model_client_factory_selects_codex_or_http_transport() -> (
    None
):
    codex_client = create_structured_model_client(_settings(mode="codex"))
    http_client = httpx.AsyncClient()
    api_client = create_structured_model_client(_settings(), http_client=http_client)

    assert isinstance(codex_client, CodexStructuredClient)
    assert isinstance(api_client, OpenAICompatibleStructuredClient)
    await http_client.aclose()


@pytest.mark.asyncio
async def test_strict_schema_fallback_validates_locally_and_records_usage() -> None:
    requests: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            return httpx.Response(
                400,
                json={"error": {"message": "response_format unsupported"}},
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"scores": [{"element_id": "target", "score": 0.7}]}
                            )
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 7,
                    "total_tokens": 19,
                },
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleStructuredClient(
        _settings(scent_reasoning_effort="low"),
        http_client=http_client,
    )
    result = await client.complete(
        CoarseScentResponse,
        (ChatMessage(role="user", content='{"goal":"Find invite"}'),),
        model="scent-model",
        role=ModelRole.COARSE_SCENT,
    )

    assert result.scores[0].score == pytest.approx(0.7)
    assert len(requests) == 2
    assert requests[0]["response_format"]["type"] == "json_schema"
    assert requests[1]["response_format"]["type"] == "json_object"
    assert requests[0]["reasoning_effort"] == "low"
    assert requests[1]["reasoning_effort"] == "low"
    assert client.records[0].endpoint_origin == "https://fake-llm.test"
    assert client.records[0].token_usage.total_tokens == 19
    assert client.records[0].attempts == 2
    assert client.manifest(ModelRole.COARSE_SCENT, "scent-model").role == (
        ModelRole.COARSE_SCENT
    )
    assert "secret-api-key" not in json.dumps(client.records[0].request)
    await http_client.aclose()


@pytest.mark.asyncio
async def test_invalid_request_falls_back_from_strict_schema() -> None:
    requests: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "code": "INVALID_REQUEST",
                        "message": "The request could not be processed.",
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"ok": true}'}}],
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleStructuredClient(_settings(), http_client=http_client)

    class SimpleResponse(BaseModel):
        ok: bool

    result = await client.complete(
        SimpleResponse,
        (ChatMessage(role="user", content="{}"),),
        model="scent-model",
        role=ModelRole.COARSE_SCENT,
    )

    assert result.ok
    assert [request["response_format"]["type"] for request in requests] == [
        "json_schema",
        "json_object",
    ]
    await http_client.aclose()


@pytest.mark.asyncio
async def test_cognitive_role_starts_in_json_object_mode() -> None:
    requests: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"ok": true}'}}],
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleStructuredClient(_settings(), http_client=http_client)

    class SimpleResponse(BaseModel):
        ok: bool

    result = await client.complete(
        SimpleResponse,
        (ChatMessage(role="user", content="{}"),),
        model="cognitive-model",
        role=ModelRole.COGNITIVE,
    )

    assert result.ok
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert "reasoning_effort" not in requests[0]
    await http_client.aclose()


@pytest.mark.asyncio
async def test_cognitive_reasoning_effort_is_forwarded_when_configured() -> None:
    requests: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok": true}'}}]},
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleStructuredClient(
        _settings(scent_reasoning_effort="low", cognitive_reasoning_effort="high"),
        http_client=http_client,
    )

    class SimpleResponse(BaseModel):
        ok: bool

    result = await client.complete(
        SimpleResponse,
        (ChatMessage(role="user", content="{}"),),
        model="cognitive-model",
        role=ModelRole.COGNITIVE,
    )

    assert result.ok
    assert requests[0]["reasoning_effort"] == "high"
    await http_client.aclose()


@pytest.mark.asyncio
async def test_invalid_structured_output_retries_only_within_bound() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        content = "not json" if calls < 3 else json.dumps({"scores": []})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleStructuredClient(_settings(), http_client=http_client)
    result = await client.complete(
        CoarseScentResponse,
        (ChatMessage(role="user", content="{}"),),
        model="scent-model",
        role=ModelRole.COARSE_SCENT,
    )

    assert result.scores == []
    assert calls == 3
    assert len(client.retry_events) == 2
    assert all(
        event.reason == "invalid-structured-output" for event in client.retry_events
    )
    await http_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "reason"),
    (
        (httpx.ConnectTimeout("connect timed out"), "connect-timeout"),
        (httpx.ReadTimeout("read timed out"), "read-timeout"),
        (httpx.ConnectError("connection failed"), "connect-error"),
        (httpx.RemoteProtocolError("server disconnected"), "protocol-error"),
    ),
)
async def test_transport_failures_record_safe_specific_category(
    error: httpx.TransportError,
    reason: str,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        raise error

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleStructuredClient(
        _settings(retry_policy={"max_attempts": 1, "base_delay_seconds": 0}),
        http_client=http_client,
    )

    with pytest.raises(ModelFailureError, match=reason):
        await client.complete(
            CoarseScentResponse,
            (ChatMessage(role="user", content="{}"),),
            model="scent-model",
            role=ModelRole.COARSE_SCENT,
        )

    assert client.records[0].attempts == 1
    await http_client.aclose()


@pytest.mark.asyncio
async def test_authentication_failure_is_terminal_without_retry() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleStructuredClient(_settings(), http_client=http_client)

    with pytest.raises(ModelFailureError, match="authentication failure"):
        await client.complete(
            CoarseScentResponse,
            (ChatMessage(role="user", content="{}"),),
            model="scent-model",
            role=ModelRole.COARSE_SCENT,
        )

    assert calls == 1
    assert client.retry_events == ()
    await http_client.aclose()


@pytest.mark.asyncio
async def test_retry_backoff_restarts_for_each_logical_model_call() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls in {1, 3}:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"scores": []})}}]},
        )

    settings = _settings(
        retry_policy={
            "max_attempts": 2,
            "base_delay_seconds": 0.25,
            "max_delay_seconds": 1.0,
            "multiplier": 2.0,
        }
    )
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleStructuredClient(settings, http_client=http_client)
    delays: list[float] = []

    async def record_sleep(delay_seconds: float) -> None:
        delays.append(delay_seconds)

    client._sleep = record_sleep  # type: ignore[method-assign]
    messages = (ChatMessage(role="user", content="{}"),)

    await client.complete(
        CoarseScentResponse,
        messages,
        model="scent-model",
        role=ModelRole.COARSE_SCENT,
    )
    await client.complete(
        CoarseScentResponse,
        messages,
        model="scent-model",
        role=ModelRole.COARSE_SCENT,
    )

    assert delays == [0.25, 0.25]
    await http_client.aclose()


@pytest.mark.asyncio
async def test_rate_limit_retry_honors_retry_after_header() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        del request
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "7"},
                json={"error": {"message": "slow down"}},
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"scores":[]}'}}]},
        )

    settings = _settings(
        retry_policy={
            "max_attempts": 2,
            "base_delay_seconds": 0.25,
            "max_delay_seconds": 1.0,
            "multiplier": 2.0,
        }
    )
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleStructuredClient(settings, http_client=http_client)
    delays: list[float] = []

    async def record_sleep(delay_seconds: float) -> None:
        delays.append(delay_seconds)

    client._sleep = record_sleep  # type: ignore[method-assign]

    await client.complete(
        CoarseScentResponse,
        (ChatMessage(role="user", content="{}"),),
        model="scent-model",
        role=ModelRole.COARSE_SCENT,
    )

    assert delays == [7.0]
    await http_client.aclose()


@pytest.mark.asyncio
async def test_shared_model_call_limiter_serializes_concurrent_requests() -> None:
    active = 0
    maximum_active = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum_active
        del request
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0)
        active -= 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"scores":[]}'}}]},
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleStructuredClient(
        _settings(retry_policy={"max_attempts": 1, "base_delay_seconds": 0}),
        http_client=http_client,
        call_limiter=asyncio.Semaphore(1),
    )
    messages = (ChatMessage(role="user", content="{}"),)

    await asyncio.gather(
        client.complete(
            CoarseScentResponse,
            messages,
            model="scent-model",
            role=ModelRole.COARSE_SCENT,
        ),
        client.complete(
            CoarseScentResponse,
            messages,
            model="scent-model",
            role=ModelRole.COARSE_SCENT,
        ),
    )

    assert maximum_active == 1
    await http_client.aclose()
