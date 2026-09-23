from uuid import UUID

import pytest

from adaptive_rag.domain.models import ConversationTurn
from adaptive_rag.memory import (
    ConversationMemoryError,
    ConversationMemoryService,
    ConversationNotFoundError,
)


class FakeConversationRepository:
    def __init__(self) -> None:
        self.histories: dict[UUID, tuple[ConversationTurn, ...]] = {}
        self.loads: list[tuple[UUID, int]] = []
        self.appends: list[dict[str, object]] = []
        self.failure: Exception | None = None

    def load(
        self,
        conversation_id: UUID,
        limit: int,
    ) -> tuple[ConversationTurn, ...] | None:
        if self.failure is not None:
            raise self.failure
        self.loads.append((conversation_id, limit))
        return self.histories.get(conversation_id)

    def append_exchange(self, **values: object) -> UUID:
        if self.failure is not None:
            raise self.failure
        self.appends.append(values)
        conversation_id = values["conversation_id"]
        assert isinstance(conversation_id, UUID)
        return conversation_id


def _service(
    repository: FakeConversationRepository,
    *,
    max_turns: int = 3,
    max_characters: int = 100,
) -> ConversationMemoryService:
    return ConversationMemoryService(
        repository,
        max_turns=max_turns,
        max_characters=max_characters,
    )


def test_new_conversation_allocates_id_without_touching_storage() -> None:
    repository = FakeConversationRepository()

    session = _service(repository).start(None)

    assert isinstance(session.conversation_id, UUID)
    assert session.history == ()
    assert session.is_new is True
    assert repository.loads == []


def test_existing_conversation_keeps_latest_contiguous_turns_within_budget() -> None:
    conversation_id = UUID("12345678-1234-5678-1234-567812345678")
    repository = FakeConversationRepository()
    repository.histories[conversation_id] = (
        ConversationTurn(1, "old-user", "old-assistant", "general_chat"),
        ConversationTurn(2, "middle-user", "middle-assistant", "general_chat"),
        ConversationTurn(3, "new-user", "new-assistant", "scientific_evidence"),
    )

    session = _service(repository, max_characters=49).start(conversation_id)

    assert repository.loads == [(conversation_id, 3)]
    assert [turn.sequence_no for turn in session.history] == [2, 3]
    assert session.is_new is False


def test_unknown_conversation_fails_explicitly() -> None:
    conversation_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")

    with pytest.raises(ConversationNotFoundError, match=str(conversation_id)):
        _service(FakeConversationRepository()).start(conversation_id)


def test_repository_failure_is_hidden_behind_memory_error() -> None:
    repository = FakeConversationRepository()
    repository.failure = RuntimeError("secret database detail")
    conversation_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")

    with pytest.raises(ConversationMemoryError, match="load conversation"):
        _service(repository).start(conversation_id)


def test_remember_forwards_idempotency_and_new_conversation_flag() -> None:
    repository = FakeConversationRepository()
    service = _service(repository)
    session = service.start(None)

    persisted_conversation_id = service.remember(
        session,
        request_id="request-1",
        user_message="hello",
        assistant_message="hi",
        response_mode="general_chat",
    )

    assert repository.appends == [
        {
            "conversation_id": session.conversation_id,
            "request_id": "request-1",
            "user_message": "hello",
            "assistant_message": "hi",
            "response_mode": "general_chat",
            "create_conversation": True,
        }
    ]
    assert persisted_conversation_id == session.conversation_id


@pytest.mark.parametrize(("max_turns", "max_characters"), [(0, 100), (1, 0)])
def test_rejects_invalid_budgets(max_turns: int, max_characters: int) -> None:
    with pytest.raises(ValueError):
        _service(
            FakeConversationRepository(),
            max_turns=max_turns,
            max_characters=max_characters,
        )
