"""Application service for bounded short-term conversational memory."""

from dataclasses import dataclass
from uuid import UUID, uuid4

from adaptive_rag.domain.models import ConversationTurn, QueryRoute
from adaptive_rag.domain.protocols import ConversationRepository


class ConversationMemoryError(RuntimeError):
    """Raised when conversation memory cannot be loaded or committed."""


class ConversationNotFoundError(ConversationMemoryError):
    """Raised when a client refers to a conversation that does not exist."""


@dataclass(frozen=True, slots=True)
class ConversationSession:
    """One request's immutable view of a conversation."""

    conversation_id: UUID
    history: tuple[ConversationTurn, ...]
    is_new: bool


class ConversationMemoryService:
    """Create/resume conversations and enforce deterministic history budgets."""

    def __init__(
        self,
        repository: ConversationRepository,
        *,
        max_turns: int,
        max_characters: int,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        if max_characters < 1:
            raise ValueError("max_characters must be positive")
        self._repository = repository
        self._max_turns = max_turns
        self._max_characters = max_characters

    def start(self, conversation_id: UUID | None) -> ConversationSession:
        """Allocate a new ID or load an existing conversation's bounded history."""

        if conversation_id is None:
            return ConversationSession(uuid4(), (), True)
        try:
            history = self._repository.load(conversation_id, self._max_turns)
        except Exception as error:
            raise ConversationMemoryError("unable to load conversation memory") from error
        if history is None:
            raise ConversationNotFoundError(f"conversation not found: {conversation_id}")
        return ConversationSession(
            conversation_id=conversation_id,
            history=_within_character_budget(history, self._max_characters),
            is_new=False,
        )

    def remember(
        self,
        session: ConversationSession,
        *,
        request_id: str,
        user_message: str,
        assistant_message: str,
        response_mode: QueryRoute,
    ) -> UUID:
        """Commit one completed exchange; repository request IDs make retries idempotent."""

        try:
            return self._repository.append_exchange(
                conversation_id=session.conversation_id,
                request_id=request_id,
                user_message=user_message,
                assistant_message=assistant_message,
                response_mode=response_mode,
                create_conversation=session.is_new,
            )
        except Exception as error:
            raise ConversationMemoryError("unable to persist conversation memory") from error


def _within_character_budget(
    history: tuple[ConversationTurn, ...],
    maximum_characters: int,
) -> tuple[ConversationTurn, ...]:
    selected: list[ConversationTurn] = []
    used = 0
    for turn in reversed(history):
        turn_size = len(turn.user_message) + len(turn.assistant_message)
        if used + turn_size > maximum_characters:
            break
        selected.append(turn)
        used += turn_size
    selected.reverse()
    return tuple(selected)
