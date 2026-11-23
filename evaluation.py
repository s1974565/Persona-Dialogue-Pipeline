import time

from pydantic import BaseModel

from service.llm import CompletionModel
from service.utilities import format_text

from clients import reasoning_for, resolve_model_client
from configurations import (
    JUDGE_REASONING,
    ConversationRecord,
    EvaluationRecord,
    ItemScore,
    Persona,
    QuestionGroup,
)


class QuantitativeResponse(BaseModel):
    answers: list[ItemScore]


def format_conversations_block(conversations: list[ConversationRecord], persona_name: str) -> str:
    blocks: list[str] = []
    for index, conversation in enumerate(conversations, start=1):
        lines = [f"\n[대화 {index}] 주제: {conversation.scenario_name}"]
        for message in conversation.messages:
            speaker = persona_name if message.role == "persona" else "인공지능"
            lines.append(f"{speaker}: {message.content}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def build_persona_sections(persona_name: str, background: str, conversation_block: str | None) -> str:
    conversation_section = ""
    if conversation_block:
        conversation_section = f"""
        [추가 대화 예시]
        (대화형 인공지능과 {persona_name}이/가 나눈 대화 기록들)
        {conversation_block}
        """
    return f"""당신은 아래 {persona_name} 페르소나를 주어진 정보를 활용해 객관적이고 정확하게 평가하여야 합니다.

    [{persona_name} 페르소나 배경]
    {background}

    {conversation_section}"""


def build_evaluation_prompt(persona_name: str, background: str, group: QuestionGroup, conversation_block: str | None) -> list[str]:
    criteria_lines = "\n".join(f"{index}. {question}" for index, question in enumerate(group.questions, start=1))

    conversation_instructions = ""
    if conversation_block:
        conversation_instructions = f"""- 페르소나 배경과 추가 대화 예시를 모두 함께 고려하세요 (대화 내용이 배경과 부합하는가 등).
        - 대화 길이 자체는 감점 요소가 아니며, 길이가 길다고 항상 좋은 것은 아닙니다. 대화의 자연스러운 흐름을 모두 고려하세요.
        - {persona_name}이/가 말한 내용만 평가에 사용하세요 (인공지능이 말한 내용은 제외)."""

    instructions = f"""
    [평가 기준(들)]
    {criteria_lines}

    [평가 지시]
    위 기준(들)에 따라서 {persona_name}의 페르소나를 정확하고 객관적으로 평가하세요.
    - 각 기준들은 서로 독립적입니다. 한 기준에 대한 답변이 다른 평가에 영향을 주면 안 됩니다.
    - 기본적으로 4점에서 시작하며, **확실한 이유**가 있을때만 점수를 적절히 가감하세요.
    {conversation_instructions}

    [답변 방법]
    각 기준에 대한 평가를 answers에 하나하나 작성해주세요.
    - number: 해당 기준의 번호 (1, 2, 3 등)
    - reasoning: 간략한 핵심 평가 근거 (강점 및 약점)
    - score: 7점 척도 평가 점수 (1점=전혀 그렇지 않다 ~ 7점=매우 그렇다)
    """
    return [format_text(build_persona_sections(persona_name, background, conversation_block)), format_text(instructions)]


def validated_item_scores(scores: list[ItemScore], group: QuestionGroup, judge: CompletionModel, persona_id: str) -> list[ItemScore]:
    expected_numbers = set(range(1, len(group.questions) + 1))
    context = f"judge {judge} on {persona_id} group {group.number}"
    score_by_number: dict[int, ItemScore] = {}
    for score in scores:
        kept = score_by_number.setdefault(score.number, score)
        if kept.score != score.score:
            raise ValueError(f"{context} scored question {score.number} twice with disagreeing scores ({kept.score}, {score.score})")
    missing_numbers = sorted(expected_numbers - set(score_by_number))
    if missing_numbers:
        raise ValueError(f"{context} did not score question(s) {missing_numbers}; the response was likely cut off")
    unknown_numbers = sorted(set(score_by_number) - expected_numbers)
    if unknown_numbers:
        raise ValueError(f"{context} scored question(s) {unknown_numbers}, which are not in the group")
    return [score_by_number[number] for number in sorted(score_by_number)]


async def evaluate_persona(
    persona: Persona,
    group: QuestionGroup,
    judge_model: CompletionModel | str,
    conversation_block: str | None = None,
    conversation_ids: list[str] | None = None,
    simulator_model: str | None = None,
    replicate_index: int = 0,
    reasoning: bool = JUDGE_REASONING,
) -> EvaluationRecord:
    judge = CompletionModel(str(judge_model))
    input_text = build_evaluation_prompt(persona.persona_name, persona.background, group, conversation_block)

    client = resolve_model_client(judge)
    usage: dict[str, int] = {}
    started_at = time.perf_counter()
    response = await client.create_single_text_response(
        input_text, model=judge, response_format=QuantitativeResponse, use_reasoning=reasoning_for(judge, reasoning), usage_sink=usage
    )
    latency_seconds = time.perf_counter() - started_at

    scores = validated_item_scores(response.answers, group, judge, persona.persona_id)
    return EvaluationRecord(
        persona_id=persona.persona_id,
        class_name=persona.class_name,
        simulator_model=simulator_model,
        judge_model=str(judge),
        group_number=group.number,
        group_name=group.name_ko,
        replicate_index=replicate_index,
        includes_conversations=bool(conversation_block),
        conversation_ids=conversation_ids,
        item_scores=scores,
        group_mean=sum(score.score for score in scores) / len(scores),
        raw_response=response.model_dump_json(),
        latency_seconds=latency_seconds,
        usage=usage or None,
    )
