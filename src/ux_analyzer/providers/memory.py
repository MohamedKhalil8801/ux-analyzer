"""Deterministic working and episodic memory policy."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ux_analyzer.domain.attention import ProgressiveObservation


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """One persona-visible memory record with deterministic retention inputs."""

    key: str
    value: str
    strength: float = 1.0
    age: int = 0
    importance: float = 0.5
    is_failure: bool = False
    viewport_id: str | None = None

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("memory key must not be empty")
        if not math.isfinite(self.strength) or not 0 <= self.strength <= 1:
            raise ValueError("memory strength must be between 0 and 1")
        if self.age < 0:
            raise ValueError("memory age must not be negative")
        if not math.isfinite(self.importance) or not 0 <= self.importance <= 1:
            raise ValueError("memory importance must be between 0 and 1")

    @classmethod
    def failure(
        cls,
        key: str,
        value: str,
        *,
        importance: float = 1.0,
        viewport_id: str | None = None,
    ) -> MemoryEntry:
        """Create an important failure record for episodic retention."""

        return cls(
            key=key,
            value=value,
            importance=importance,
            is_failure=True,
            viewport_id=viewport_id,
        )

    @property
    def retention_score(self) -> float:
        """Return base score used for deterministic ordering."""

        return self.strength * self.importance


@dataclass(frozen=True, slots=True)
class MemoryState:
    """Immutable working and episodic memory collections."""

    working: tuple[MemoryEntry, ...] = ()
    episodic: tuple[MemoryEntry, ...] = ()

    def __post_init__(self) -> None:
        working = tuple(self.working)
        episodic = tuple(self.episodic)
        for name, entries in (("working", working), ("episodic", episodic)):
            keys = [entry.key for entry in entries]
            if len(keys) != len(set(keys)):
                raise ValueError(f"duplicate {name} memory key")
        object.__setattr__(self, "working", working)
        object.__setattr__(self, "episodic", episodic)

    @classmethod
    def empty(cls) -> MemoryState:
        return cls()

    @property
    def working_memory(self) -> tuple[MemoryEntry, ...]:
        return self.working

    @property
    def episodic_memory(self) -> tuple[MemoryEntry, ...]:
        return self.episodic


@dataclass(frozen=True, slots=True)
class MemoryPolicyConfig:
    """Versioned formulas and capacities for deterministic memory behavior."""

    version: str = "memory-policy-v1"
    working_capacity: int = 3
    episodic_capacity: int = 20
    decay_rate: float = 0.1
    failure_retention_bonus: float = 1.0
    failure_retention_floor: float = 0.0

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("memory policy version must not be empty")
        if self.working_capacity <= 0:
            raise ValueError("working_capacity must be greater than zero")
        if self.episodic_capacity <= 0:
            raise ValueError("episodic_capacity must be greater than zero")
        if not math.isfinite(self.decay_rate) or not 0 <= self.decay_rate <= 1:
            raise ValueError("decay_rate must be between 0 and 1")
        if not math.isfinite(self.failure_retention_bonus) or (
            self.failure_retention_bonus < 0
        ):
            raise ValueError("failure_retention_bonus must not be negative")
        if (
            not math.isfinite(self.failure_retention_floor)
            or not 0 <= (self.failure_retention_floor) <= 1
        ):
            raise ValueError("failure_retention_floor must be between 0 and 1")


class MemoryPolicy:
    """Apply deterministic decay, capacity, and failure-retention rules."""

    def __init__(self, config: MemoryPolicyConfig | None = None) -> None:
        self.config = config or MemoryPolicyConfig()

    def update(
        self,
        memory: MemoryState | None = None,
        entries: Iterable[MemoryEntry] = (),
        *,
        working: Iterable[MemoryEntry] = (),
        episodic: Iterable[MemoryEntry] = (),
        failures: Iterable[MemoryEntry] = (),
        observation: ProgressiveObservation | None = None,
        elapsed_steps: int = 1,
    ) -> MemoryState:
        """Return next memory state using explicit, seed-independent formulas."""

        if elapsed_steps < 0:
            raise ValueError("elapsed_steps must not be negative")
        current = memory or MemoryState.empty()
        working_entries = list(current.working)
        episodic_entries = list(current.episodic)
        working_entries = [
            self._decay(entry, elapsed_steps) for entry in working_entries
        ]
        episodic_entries = [
            self._decay(entry, elapsed_steps) for entry in episodic_entries
        ]

        new_working = list(entries)
        new_working.extend(working)
        if observation is not None:
            new_working.extend(self._entries_from_observation(observation))
        new_episodic = list(episodic)
        new_episodic.extend(failures)

        working_state = self._merge_recent(working_entries, new_working)
        episodic_state = self._merge_ranked(episodic_entries, new_episodic)
        return MemoryState(working=working_state, episodic=episodic_state)

    def recall(
        self,
        memory: MemoryState,
        keys: Iterable[str] | str | None = None,
        *,
        include_episodic: bool = True,
    ) -> tuple[MemoryEntry, ...]:
        """Recall records in stable working-then-episodic order."""

        entries = list(memory.working)
        if include_episodic:
            entries.extend(memory.episodic)
        if keys is None:
            return tuple(entries)
        wanted = {keys} if isinstance(keys, str) else set(keys)
        return tuple(entry for entry in entries if entry.key in wanted)

    def _decay(self, entry: MemoryEntry, elapsed_steps: int) -> MemoryEntry:
        """Apply `strength * (1 - decay_rate) ** steps`, with failure floor."""

        strength = entry.strength * (1 - self.config.decay_rate) ** elapsed_steps
        if entry.is_failure:
            strength = max(strength, self.config.failure_retention_floor)
        return MemoryEntry(
            key=entry.key,
            value=entry.value,
            strength=strength,
            age=entry.age + elapsed_steps,
            importance=entry.importance,
            is_failure=entry.is_failure,
            viewport_id=entry.viewport_id,
        )

    def _merge_recent(
        self,
        existing: list[MemoryEntry],
        additions: Iterable[MemoryEntry],
    ) -> tuple[MemoryEntry, ...]:
        by_key: dict[str, MemoryEntry] = {entry.key: entry for entry in existing}
        order = [entry.key for entry in existing]
        for entry in additions:
            if entry.key in by_key:
                order.remove(entry.key)
            by_key[entry.key] = entry
            order.append(entry.key)
        return tuple(by_key[key] for key in order[-self.config.working_capacity :])

    def _merge_ranked(
        self,
        existing: list[MemoryEntry],
        additions: Iterable[MemoryEntry],
    ) -> tuple[MemoryEntry, ...]:
        by_key: dict[str, MemoryEntry] = {entry.key: entry for entry in existing}
        for entry in additions:
            by_key[entry.key] = entry
        ranked = sorted(
            by_key.values(),
            key=lambda entry: (
                entry.is_failure,
                entry.strength * entry.importance
                + (self.config.failure_retention_bonus if entry.is_failure else 0),
                -entry.age,
                entry.key,
            ),
            reverse=True,
        )
        return tuple(ranked[: self.config.episodic_capacity])

    @staticmethod
    def _entries_from_observation(
        observation: ProgressiveObservation,
    ) -> tuple[MemoryEntry, ...]:
        return tuple(
            MemoryEntry(
                key=element.id,
                value=element.label,
                viewport_id=observation.viewport_id,
            )
            for element in (
                *observation.remembered_elements,
                *observation.newly_revealed_elements,
            )
        )


def memory_from_mapping(payload: Mapping[str, str]) -> MemoryState:
    """Create working records from simple fixture-like key/value data."""

    return MemoryState(
        working=tuple(
            MemoryEntry(key=key, value=value) for key, value in payload.items()
        )
    )
