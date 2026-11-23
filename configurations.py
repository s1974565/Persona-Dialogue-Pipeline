from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, BaseModel, Field, model_validator

from service.llm import CompletionModel
from service.utilities import now, uuid

########## Paths ##########

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "results"

PERSONA_DIR = DATA_DIR / "persona"
SEED_DUMP_PATH = DATA_DIR / "seeds.txt"
SCENARIOS_PATH = DATA_DIR / "scenarios.txt"
PARTNER_PROMPT_PATH = DATA_DIR / "partner.txt"
PPS_QUESTIONS_PATH = DATA_DIR / "pps_questions.txt"
CHARACTER_LLM_QUESTIONS_PATH = DATA_DIR / "character_llm_questions.txt"

########## Models ##########

SIMULATOR_MODELS: list[CompletionModel] = [
    CompletionModel.GPT_5_5,
    CompletionModel.GPT_5_4_MINI,
    CompletionModel.CLAUDE_SONNET_4_6,
    CompletionModel.CLAUDE_HAIKU_4_5,
    CompletionModel.GEMINI_3_5_FLASH,
    CompletionModel.GEMINI_3_1_FLASH_LITE,
    CompletionModel.DEEPSEEK_V4_FLASH,
    CompletionModel.QWEN_3_5_397B_A17B,
]

ABLATION_SIMULATOR_MODELS: list[CompletionModel] = [
    CompletionModel.GPT_5_5,
    CompletionModel.CLAUDE_SONNET_4_6,
    CompletionModel.GEMINI_3_5_FLASH,
]

# All three are also simulators.
JUDGE_MODELS: list[CompletionModel] = [
    CompletionModel.GPT_5_5,
    CompletionModel.CLAUDE_SONNET_4_6,
    CompletionModel.GEMINI_3_5_FLASH,
]

PARTNER_MODEL: CompletionModel = CompletionModel.GPT_5_4_MINI  # also a simulator, so its own conversations are self-play
PARTNER_NAME = "인공지능"  # how the persona is told to address its partner; an expert's job title in Study 2

MODELS_WITHOUT_REASONING_TOGGLE: set[CompletionModel] = {CompletionModel.CLAUDE_HAIKU_4_5}

CONVERSATION_REASONING = False
JUDGE_REASONING = True
RESTORATION_REASONING = False

########## Persona sampling ##########

INITIAL_SAMPLES_PER_CLASS = 4
ADDITIONAL_SAMPLES_PER_CLASS = 6
SAMPLING_SEED = 20240614  # the second draw uses SAMPLING_SEED + 1

########## Arms ##########

FREE_TOPIC_SCENARIO_ID = "scenario_6"  # fixes no topics by turn, so only the free-turn arm includes it
SEED_ONLY_LIVING_ARRANGEMENT = "- 혼자서 살고 있음 (독거노인)"

ARMS: dict[str, dict] = {
    "fixed_15turn": {
        "output_subdir": "fixed_15turn",
        "simulators": SIMULATOR_MODELS,
        "includes_free_topic_scenario": False,
        "is_minimal_persona_prompt": False,
        "allow_early_termination": False,
        "is_seed_only": False,
    },
    "free_turn": {
        "output_subdir": "free_turn",
        "simulators": SIMULATOR_MODELS,
        "includes_free_topic_scenario": True,
        "is_minimal_persona_prompt": False,
        "allow_early_termination": True,
        "is_seed_only": False,
    },
    "background_minimal": {
        "output_subdir": "ablation/background_minimal",
        "simulators": ABLATION_SIMULATOR_MODELS,
        "includes_free_topic_scenario": False,
        "is_minimal_persona_prompt": True,
        "allow_early_termination": False,
        "is_seed_only": False,
    },
    "seed_minimal": {
        "output_subdir": "ablation/seed_minimal",
        "simulators": ABLATION_SIMULATOR_MODELS,
        "includes_free_topic_scenario": False,
        "is_minimal_persona_prompt": True,
        "allow_early_termination": False,
        "is_seed_only": True,
    },
}
DEFAULT_ARM = "fixed_15turn"

########## Conversation ##########

MAX_TURNS = 15
UTTERANCE_CANDIDATE_COUNT = 5
LOWEST_CANDIDATE_DROP_COUNT = 2
PERSONA_TURN_MAX_ATTEMPTS = 5  # some simulators return fewer candidates than requested

########## Evaluation ##########

PPS_SCORE_MINIMUM = 1
PPS_SCORE_MAXIMUM = 7
EVALUATION_REPLICATES = 3
PPS_LLM_JUDGE_EXCLUDED_GROUPS: set[str] = {"유사성"}  # an LLM rater cannot be similar to the persona; the value is the group name in pps_questions.txt
CHARACTER_LLM_GROUP_NUMBER = 7
CHARACTER_LLM_GROUP_NAME = "Character LLM"  # scored only when the conversations are supplied

RESTORATION_REPLICATES = 1

########## Concurrency ##########

CONVERSATION_CONCURRENCY = 8
EVALUATION_PAIRS_IN_FLIGHT = 5
EVALUATION_PAIR_CONCURRENCY = 8
RESTORATION_CONCURRENCY = 6

########## Schema ##########


class StrippedBaseModel(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def strip_string_fields(cls, values):
        if not isinstance(values, dict):
            return values

        for field_key, field_value in values.items():
            if isinstance(field_value, str):
                values[field_key] = field_value.strip()

            elif isinstance(field_value, list):
                normalised_items: list = []
                for item in field_value:
                    if isinstance(item, str):
                        stripped_item = item.strip()
                        if stripped_item:
                            normalised_items.append(stripped_item)
                    else:
                        normalised_items.append(item)

                values[field_key] = normalised_items

        return values


class Persona(BaseModel):
    persona_id: str
    persona_name: str
    background: str
    class_name: str | None = None


class ScenarioTopic(StrippedBaseModel):
    description: str
    injection_turn: Annotated[int, Field(ge=1)] = 1


class ScenarioContent(StrippedBaseModel):
    scenario_id: str
    scenario_name: str
    topics: Annotated[list[ScenarioTopic], Field(min_length=1, max_length=5)]

    @model_validator(mode="after")
    def validate_injection_turns(self):
        if self.topics[0].injection_turn != 1:
            raise ValueError("The first topic's injection_turn must be 1.")
        for previous_topic, current_topic in zip(self.topics, self.topics[1:]):
            if current_topic.injection_turn <= previous_topic.injection_turn:
                raise ValueError("Topic injection_turn values must be strictly increasing.")
        return self


class QuestionGroup(BaseModel):
    number: int
    name_ko: str
    name_en: str
    questions: list[str]


class PersonaUtteranceCandidate(StrippedBaseModel):
    utterance: str
    reasoning: str
    probability: float


class PersonaTurnResponse(StrippedBaseModel):
    mindset: str
    utterance_list: list[PersonaUtteranceCandidate]


class PersonaTurnResponseWithQuit(PersonaTurnResponse):
    quit_dialogue: bool = False


class PersonaTurnResponseMinimal(StrippedBaseModel):
    utterance: str


class ConversationMessage(StrippedBaseModel):
    role: Literal["persona", "partner"]
    content: str
    mindset: str | None = None
    reasoning: str | None = None
    utterance_candidates: list[PersonaUtteranceCandidate] | None = None
    topic_injection: ScenarioTopic | None = None


class ItemScore(BaseModel):
    number: int
    reasoning: str
    score: Annotated[int, Field(ge=PPS_SCORE_MINIMUM, le=PPS_SCORE_MAXIMUM)]


########## Records ##########


class RecordBase(BaseModel):
    record_id: str = Field(default_factory=uuid)
    created_at: AwareDatetime = Field(default_factory=now)


class ConversationRecord(RecordBase):
    persona_id: str
    class_name: str | None = None
    simulator_model: str
    partner_model: str
    scenario_id: str
    scenario_name: str
    turn_count: int
    total_character_count: int
    messages: list[ConversationMessage]
    terminated_by_persona: bool = False
    reached_turn_cap: bool = False
    allow_early_termination: bool = False
    is_minimal_persona_prompt: bool = False


class ItemAnswer(BaseModel):
    item_key: str
    raw_answer: dict[str, Any]
    healthy_flag: int
    reasoning: str | None = None
    latency_seconds: float = 0.0


class RestorationRecord(RecordBase):
    persona_id: str
    true_class: str
    simulator_model: str
    replicate_index: int
    item_answers: list[ItemAnswer]
    healthy_code: dict[str, int]
    posterior: dict[str, float]
    posterior_vector: list[float]
    predicted_class: str
    cross_entropy: float
    true_class_mass: float
    correct: bool


class EvaluationRecord(RecordBase):
    persona_id: str
    class_name: str | None = None
    simulator_model: str | None = None
    judge_model: str
    group_number: int
    group_name: str
    replicate_index: int
    includes_conversations: bool = False
    conversation_ids: list[str] | None = None
    item_scores: list[ItemScore]
    group_mean: float
    raw_response: str
    latency_seconds: float = 0.0
    usage: dict[str, int] | None = None
