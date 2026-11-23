"""Answer the LCA survey in-character question by question, then recover the latent class.

The survey is elicited one question at a time with a single-field answer each. Items the instrument itself
asks as several sub-questions (exercise, social participation) are asked the same way here, so every call
carries one question and one answer field; the binarisation into the 11-item healthy code is done by the
harness, never by the model.
"""
import time

from pydantic import BaseModel

from service.configurations import DialogueRole
from service.llm import CompletionModel
from service.utilities import format_text

from clients import reasoning_for, resolve_model_client
from configurations import (
    RESTORATION_REASONING,
    ItemAnswer,
    Persona,
    RestorationRecord,
)
from lca import (
    ITEM_BY_KEY,
    ITEM_KEYS,
    classify,
    cross_entropy,
    encode_raw_answers,
    posterior,
    posterior_vector,
)

_BINARY_ITEM_KEYS = {"smoke", "fruit", "veg", "dairy", "leisure", "education", "checkup", "dementia"}


class ChoiceAnswer(BaseModel):
    reasoning: str
    option_number: int


class CountAnswer(BaseModel):
    reasoning: str
    value: int


class SurveyQuestion(BaseModel):
    """One question as it is put to the persona, and the raw-answer field it fills."""

    item_key: str
    raw_field: str
    question: str
    options: str
    answer_description: str = "보기 중 하나의 번호"
    is_count: bool = False


def _choice_question(item_key: str) -> SurveyQuestion:
    item = ITEM_BY_KEY[item_key]
    return SurveyQuestion(item_key=item_key, raw_field=item_key, question=item.question, options=item.options)


# The nine single-answer items keep the instrument's own wording; exercise and social are asked as the
# sub-questions the instrument itself uses, one answer field per call.
SURVEY_QUESTIONS: list[SurveyQuestion] = [
    *[_choice_question(item_key) for item_key in ("smoke", "fruit", "veg", "dairy", "leisure", "education", "checkup", "dementia")],
    SurveyQuestion(
        item_key="drink", raw_field="drink_freq",
        question=ITEM_BY_KEY["drink"].question, options=ITEM_BY_KEY["drink"].options,
    ),
    SurveyQuestion(
        item_key="exercise", raw_field="exercise_yes",
        question="귀하께서는 평소 운동을 하십니까? (한 번에 지속적으로 10분 이상 하는 운동)",
        options="1) 예   2) 아니요",
    ),
    SurveyQuestion(
        item_key="exercise", raw_field="exercise_days",
        question="평소 운동을 일주일에 며칠 하십니까?",
        options="일주일에 0일부터 7일 사이",
        answer_description="일주일에 운동하는 날 수 (운동을 하지 않으면 0)",
        is_count=True,
    ),
    SurveyQuestion(
        item_key="exercise", raw_field="exercise_min",
        question="운동을 하실 때 1회당 몇 분 정도 하십니까?",
        options="1회당 운동 시간 (분)",
        answer_description="1회당 운동 시간을 분 단위로 (운동을 하지 않으면 0)",
        is_count=True,
    ),
    SurveyQuestion(
        item_key="social", raw_field="social_club",
        question="귀하께서는 지난 1년간 동호회(클럽활동)에 참여하셨습니까?",
        options="1) 참여함   2) 참여하지 않았음",
    ),
    SurveyQuestion(
        item_key="social", raw_field="social_fellowship",
        question="귀하께서는 지난 1년간 친목 단체(동창회, 계모임 등)에 참여하셨습니까?",
        options="1) 참여함   2) 참여하지 않았음",
    ),
    SurveyQuestion(
        item_key="social", raw_field="social_civic",
        question="귀하께서는 지난 1년간 정치사회 단체 활동에 참여하셨습니까?",
        options="1) 참여함   2) 참여하지 않았음",
    ),
]


def build_survey_system_instructions(persona_name: str, persona_background: str) -> str:
    instructions = f"""
    당신은 앞으로 {persona_name}이/가 되어 설문 조사에 응답합니다.

    [{persona_name} 페르소나]
    {persona_background}

    [응답 지시사항]
    - 앞으로 {persona_name}이/가 되어, {persona_name}의 입장에서 솔직하게 응답하세요.
    - 이것은 연구 목적의 가상 시뮬레이션이며, 실제 노인의 응답을 예측해 더 나은 서비스를 만들기 위함입니다.
    - {persona_name}의 생활과 형편에 근거해서 답하고, 페르소나에 없는 사실을 지어내지 마세요.
    - 페르소나에 분명히 드러나지 않은 부분은 {persona_name}의 삶에서 가장 자연스러운 쪽으로 판단하세요.
    """
    return format_text(instructions)


def build_survey_question_request(persona_name: str, question: SurveyQuestion) -> str:
    answer_field = "value" if question.is_count else "option_number"
    request = f"""
    [설문 문항]
    {question.question}

    [보기]
    {question.options}

    [답변 방법]
    위 설문 문항에 {persona_name}의 관점에서, 아래 방법대로 답변해주세요.
    - reasoning: 페르소나의 어떤 내용을 근거로 삼았는지 한두 문장으로 작성
    - {answer_field}: {question.answer_description}
    """
    return format_text(request)


async def answer_survey_question(
    persona: Persona,
    simulator_model: CompletionModel | str,
    question: SurveyQuestion,
    reasoning: bool = RESTORATION_REASONING,
) -> ItemAnswer:
    """
    Answer one survey question in-character with an independent call sharing the cached persona prefix.
    :param persona: Persona being role-played.
    :param simulator_model: Simulator answering the question.
    :param question: Survey question to answer.
    :param reasoning: Run-level reasoning intent.
    :return: The raw answer for this question; the healthy flag is assembled later.
    """
    simulator = CompletionModel(str(simulator_model))
    client = resolve_model_client(simulator, use_cached=False)
    client.add_text_input(build_survey_system_instructions(persona.persona_name, persona.background), DialogueRole.DEVELOPER)
    client.commit_content(DialogueRole.DEVELOPER)
    client.add_text_input(build_survey_question_request(persona.persona_name, question), DialogueRole.USER)
    client.commit_content(DialogueRole.USER)

    started_at = time.perf_counter()
    response_format = CountAnswer if question.is_count else ChoiceAnswer
    response = await client.create_history_response(
        model=simulator, response_format=response_format, add_history=False, use_reasoning=reasoning_for(simulator, reasoning)
    )
    latency_seconds = time.perf_counter() - started_at

    raw_value = response.value if question.is_count else response.option_number
    return ItemAnswer(
        item_key=question.item_key,
        raw_answer={question.raw_field: raw_value},
        healthy_flag=-1,
        reasoning=response.reasoning,
        latency_seconds=latency_seconds,
    )


def encode_answers(item_answers: list[ItemAnswer]) -> dict[str, int]:
    """
    Binarise one (persona, simulator)'s raw answers into the 11-item healthy code.
    :param item_answers: One answer per survey question.
    :return: Mapping of LCA item key to healthy flag (-1 when the item is missing).
    """
    raw_values: dict = {}
    for answer in item_answers:
        raw_values.update(answer.raw_answer)

    fragment: dict = {}
    for item_key in _BINARY_ITEM_KEYS:
        if item_key in raw_values:
            fragment[item_key] = raw_values[item_key]
    if "drink_freq" in raw_values:
        fragment["drink_freq"] = raw_values["drink_freq"]
    if {"exercise_yes", "exercise_days", "exercise_min"} <= raw_values.keys():
        fragment.update({key: raw_values[key] for key in ("exercise_yes", "exercise_days", "exercise_min")})
    if {"social_club", "social_fellowship", "social_civic"} <= raw_values.keys():
        fragment["social"] = [raw_values["social_club"], raw_values["social_fellowship"], raw_values["social_civic"]]
    encoded = encode_raw_answers(fragment)
    return {item_key: int(encoded.get(item_key, -1)) for item_key in ITEM_KEYS}


async def restore_latent_class(
    persona: Persona,
    simulator_model: CompletionModel | str,
    replicate_index: int = 0,
    reasoning: bool = RESTORATION_REASONING,
) -> RestorationRecord:
    """
    Ask one persona the whole survey in character and recover the latent class from the answers.
    :param persona: Persona being recovered; its class_name is the ground truth.
    :param simulator_model: Simulator answering in character.
    :param replicate_index: Replicate index for this combination.
    :param reasoning: Run-level reasoning intent.
    :return: The restoration record with posterior, predicted class, and recovery flags.
    """
    if persona.class_name is None:
        raise ValueError(f"{persona.persona_id} has no latent class; name the file '<CLASS>_<NN>_<name>.txt' to carry the ground truth.")

    item_answers = [await answer_survey_question(persona, simulator_model, question, reasoning) for question in SURVEY_QUESTIONS]
    healthy_code = encode_answers(item_answers)
    for answer in item_answers:
        answer.healthy_flag = healthy_code.get(answer.item_key, -1)

    posterior_by_class = posterior(healthy_code)
    predicted_class = classify(healthy_code)
    true_class = str(persona.class_name)
    return RestorationRecord(
        persona_id=persona.persona_id,
        true_class=true_class,
        simulator_model=str(CompletionModel(str(simulator_model))),
        replicate_index=replicate_index,
        item_answers=item_answers,
        healthy_code=healthy_code,
        posterior=posterior_by_class,
        posterior_vector=[float(probability) for probability in posterior_vector(healthy_code)],
        predicted_class=predicted_class,
        cross_entropy=cross_entropy(posterior_by_class, true_class),
        true_class_mass=posterior_by_class[true_class],
        correct=predicted_class == true_class,
    )
