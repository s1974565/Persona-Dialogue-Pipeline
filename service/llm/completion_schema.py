from abc import ABC, abstractmethod
from typing import Type, Any

from service.configurations import DialogueRole
from .configurations import (
    LLMProvider,
    CompletionModel,
    CompletionResponseFormat,
    ReasoningEffort,
)
from .prompt_tracker import PromptTracker


class CompletionSchema(ABC):
    def __init__(self):
        self.prompt_tracker = PromptTracker()

        # Chat completion API is designed for alternating turns.
        # Consecutive messages from the same role should be combined into a single content.
        self.current_user_content = []
        self.current_developer_content = []
        self.current_assistant_content = []
        self.dialogue_record = []

    @property
    @abstractmethod
    def provider(self) -> LLMProvider:
        pass

    ########## Dialogue Record ##########

    def clear_current_content(self, role: DialogueRole) -> None:
        if role == DialogueRole.USER:
            self.current_user_content = []
        elif role == DialogueRole.DEVELOPER:
            self.current_developer_content = []
        elif role == DialogueRole.ASSISTANT:
            self.current_assistant_content = []
        else:
            raise ValueError(f"Unsupported dialogue role: {role}")

    def add_text_input(self, text: str, role: DialogueRole) -> None:
        """
        Adding a text input to the given role's current content.
        The content block is not yet committed to the conversation history.
        :param text:
        :param role:
        :return:
        """
        message = {
            "type": "text",
            "text": text,
        }

        if role == DialogueRole.USER:
            self.current_user_content.append(message)
        elif role == DialogueRole.DEVELOPER:
            self.current_developer_content.append(message)
        elif role == DialogueRole.ASSISTANT:
            self.current_assistant_content.append(message)
        else:
            raise ValueError(f"Unsupported dialogue role: {role}")

    def commit_content(self, role: DialogueRole) -> None:
        """
        Commit the given role's current content to the dialogue history and clears the content.
        :param role:
        :return:
        """
        if self.dialogue_record and self.dialogue_record[-1]["role"] == role:
            raise ValueError(f"Cannot commit consecutive {role.value} messages. Chat completion API requires alternating turns.")

        if role == DialogueRole.USER:
            if self.current_user_content:
                self.dialogue_record.append(
                    {
                        "role": role,
                        "content": self.current_user_content,
                    }
                )
                self.clear_current_content(role)
        elif role == DialogueRole.DEVELOPER:
            if self.current_developer_content:
                self.dialogue_record.append(
                    {
                        "role": role,
                        "content": self.current_developer_content,
                    }
                )
                self.clear_current_content(role)
        elif role == DialogueRole.ASSISTANT:
            if self.current_assistant_content:
                self.dialogue_record.append(
                    {
                        "role": role,
                        "content": self.current_assistant_content,
                    }
                )
                self.clear_current_content(role)
        else:
            raise ValueError(f"Unsupported dialogue role: {role}")

    ########## Response Creation ##########

    @abstractmethod
    def _build_api_messages(self, dialogue_history: list[dict]) -> tuple[list[dict], str]:
        """
        Transforms the dialogue history into the provider-specific API message format.
        :param dialogue_history: Dialogue history (list of role/content dicts).
        :return:
        """
        pass

    def _extra_completion_kwargs(self, model: CompletionModel, use_reasoning: bool) -> dict[str, Any]:
        return {}

    def _resolve_reasoning_effort(self, model: CompletionModel, options: list[ReasoningEffort], use_reasoning: bool) -> ReasoningEffort:
        """
        Resolve the configured effort for the requested reasoning state.
        :param model: Model being resolved, used for the error message.
        :param options: Effort options ordered [reasoning-off, reasoning-on]; UNSUPPORTED marks an unavailable state.
        :param use_reasoning: Whether to select the reasoning-on effort.
        :return: The resolved effort; DISABLED means reasoning off with no effort level.
        """
        option = options[1] if use_reasoning else options[0]
        if option == ReasoningEffort.UNSUPPORTED:
            raise ValueError(f"{model} does not support reasoning {'enabled' if use_reasoning else 'disabled'}.")
        return option

    @abstractmethod
    async def create_single_history_response(
        self,
        dialogue_history: list[dict],
        model: CompletionModel | None = None,
        response_format: Type[CompletionResponseFormat] | None = None,
        use_reasoning: bool = True,
        usage_sink: dict | None = None,
    ) -> str | CompletionResponseFormat:
        """
        Generates a response with auto-retries and adaptive model selection. The response is not added to the history.
        This function has two main features for robustness:
        1. API Retries: Automatically retries on transient API failures (e.g., rate limits) with exponential backoff.
        2. Model Escalation: If called repeatedly with the same dialogue_history (signaling that the prior response was inadequate),
                             it automatically upgrades from a basic model to a superior one. This is only active when the model is not specified.
        :param dialogue_history: Conversational history.
        :param model: Model to use. If None, the model is selected adaptively based on the number of calls for the same prompt.
        :param response_format: Optional pydantic.BaseModel to structure the JSON output.
        :param use_reasoning: If True, use the model's configured reasoning effort; if False, disable reasoning where supported.
        :return: Generated response either in text or the given pydantic model.
        """
        pass

    async def create_single_text_response(
        self,
        input_text: str | list[str],
        model: CompletionModel | None = None,
        role: DialogueRole = DialogueRole.USER,
        response_format: Type[CompletionResponseFormat] | None = None,
        use_reasoning: bool = True,
        usage_sink: dict | None = None,
    ) -> str | CompletionResponseFormat:
        """
        Generates a response for a single text input. The response is not added to the history.
        :param input_text: Input text or list of input text. List items are added as separate contents (better for prompt caching).
        :param model: Model to use. If None, the model is selected adaptively.
        :param role: Role of the input message.
        :param response_format: Optional pydantic.BaseModel to structure the JSON output.
        :param use_reasoning: If True, use the model's configured reasoning effort; if False, disable reasoning where supported.
        :param usage_sink: Optional dict populated with this call's token usage. Per-call so concurrent calls on a shared client never race.
        :return: Generated response either in text or the given pydantic model.
        """
        if isinstance(input_text, str):
            input_text = [input_text]
        if not input_text:
            raise ValueError("Input text cannot be empty")

        user_content = []
        for text in input_text:
            text_entry = {
                "type": "text",
                "text": text,
            }
            user_content.append(text_entry)

        dialogue_history = [
            {
                "role": role,
                "content": user_content,
            }
        ]

        response = await self.create_single_history_response(dialogue_history, model, response_format, use_reasoning=use_reasoning, usage_sink=usage_sink)
        return response

    async def create_history_response(
        self,
        model: CompletionModel | None = None,
        response_format: Type[CompletionResponseFormat] | None = None,
        add_history: bool = True,
        use_reasoning: bool = True,
    ) -> str | CompletionResponseFormat:
        """
        Generates a response with the current dialogue history.
        :param model: The model to use. If None, the model is selected adaptively.
        :param response_format: Optional pydantic.BaseModel to structure the JSON output.
        :param add_history: If True, the response is added to the dialogue history.
        :param use_reasoning: If True, use the model's configured reasoning effort; if False, disable reasoning where supported.
        :return: Generated response.
        :raises LLMAPIError: If all API retries fail.
        """
        response = await self.create_single_history_response(self.dialogue_record, model, response_format, use_reasoning=use_reasoning)

        if add_history:
            if response_format is None:
                assistant_content = response
            else:
                assistant_content = response.model_dump_json()

            self.add_text_input(assistant_content, DialogueRole.ASSISTANT)
            self.commit_content(DialogueRole.ASSISTANT)

        return response
