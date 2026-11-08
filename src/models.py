"""Conversation state with validated inputs and detached serialization.

System instructions are assembled by the answer service, not defined here.
Workflow status strings remain extensible for existing callers.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label} must be a nonempty string')
    return value


def _timestamp(value: Any, label: str) -> str:
    _text(value, label)
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError(f'{label} must be an ISO timestamp') from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f'{label} must include a timezone')
    return parsed.astimezone(timezone.utc).isoformat()


@dataclass
class CaseState:
    messages: list[dict[str, str]] = field(default_factory=list)
    scenario: str | None = None
    collected_facts: dict[str, Any] = field(default_factory=dict)
    missing_facts: list[str] = field(default_factory=list)
    workflow_status: str = 'not_started'
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not isinstance(self.messages, list):
            raise ValueError('messages must be a list')
        for message in self.messages:
            if not isinstance(message, dict):
                raise ValueError('Each message must be a dictionary')
            self._validate_message(message.get('role'), message.get('content'))
        if not isinstance(self.collected_facts, dict):
            raise ValueError('collected_facts must be a dictionary')
        for key in self.collected_facts:
            _text(key, 'Fact key')
        if not isinstance(self.missing_facts, list):
            raise ValueError('missing_facts must be a list')
        for fact in self.missing_facts:
            _text(fact, 'Missing fact')
        if self.scenario is not None:
            _text(self.scenario, 'scenario')
        _text(self.workflow_status, 'workflow_status')
        self.created_at = _timestamp(self.created_at, 'created_at')
        self.updated_at = _timestamp(self.updated_at, 'updated_at')
        self.messages = deepcopy(self.messages)
        self.collected_facts = deepcopy(self.collected_facts)
        self.missing_facts = list(self.missing_facts)

    @staticmethod
    def _validate_message(role: Any, content: Any) -> None:
        if not isinstance(role, str) or role not in {'user', 'assistant', 'system'}:
            raise ValueError(f'Invalid message role: {role!r}')
        _text(content, 'Message content')

    def add_message(self, role: str, content: str) -> None:
        self._validate_message(role, content)
        self.messages.append({'role': role, 'content': content})
        self.updated_at = utc_now()

    def prior_messages(self, current_user_text: str) -> list[dict[str, str]]:
        """Return a detached conversation prefix for the current turn."""
        messages = self.messages
        if (messages and messages[-1]['role'] == 'user'
                and messages[-1]['content'] == current_user_text):
            messages = messages[:-1]
        return deepcopy(messages)

    def update_fact(self, key: str, value: Any) -> None:
        _text(key, 'Fact key')
        self.collected_facts[key] = deepcopy(value)
        self.updated_at = utc_now()

    def set_scenario(self, scenario: str) -> None:
        _text(scenario, 'scenario')
        self.scenario = scenario
        self.workflow_status = 'collecting_facts'
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return deepcopy({
            'messages': self.messages, 'scenario': self.scenario,
            'collected_facts': self.collected_facts, 'missing_facts': self.missing_facts,
            'workflow_status': self.workflow_status,
            'created_at': self.created_at, 'updated_at': self.updated_at,
        })

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> 'CaseState':
        if not isinstance(value, dict):
            raise ValueError('Saved case state must be a dictionary')
        names = ('messages', 'scenario', 'collected_facts', 'missing_facts',
                 'workflow_status', 'created_at', 'updated_at')
        return cls(**{name: value[name] for name in names if name in value})
