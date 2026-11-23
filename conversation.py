import random

from service.configurations import DialogueRole
from service.llm import CompletionModel
from service.utilities import format_text

from clients import reasoning_for, resolve_model_client
from configurations import (
    CONVERSATION_REASONING,
    LOWEST_CANDIDATE_DROP_COUNT,
    MAX_TURNS,
    PARTNER_MODEL,
    PARTNER_NAME,
    PERSONA_TURN_MAX_ATTEMPTS,
    UTTERANCE_CANDIDATE_COUNT,
    ConversationMessage,
    ConversationRecord,
    Persona,
    PersonaTurnResponse,
    PersonaTurnResponseMinimal,
    PersonaTurnResponseWithQuit,
    PersonaUtteranceCandidate,
    ScenarioContent,
)
from resources import load_partner_system_prompt


def build_persona_system_instructions(persona_name: str, persona_background: str, utterance_candidate_count: int, partner_name: str = PARTNER_NAME, allow_early_termination: bool = False) -> str:
    termination_section = ""
    if allow_early_termination:
        termination_section = f"""
    [대화 종료]
    {persona_name}의 관점에서, 더 이상 대화를 이어나가고 싶지 않다면 quit_dialogue 를 True로 설정하세요.
    (예: 대화가 재미없음, 피곤하거나 지침, 다른 할 일이 있음, 더 할 이야기가 없음, 자연스러운 작별 등)
    대화를 종료할때 {persona_name}에게는 아무런 불이익이 없습니다.
    """
    instructions = f"""
    당신은 앞으로 {persona_name}이/가 되어 {partner_name}과/와 이야기합니다.

    [{persona_name} 페르소나]
    {persona_background}

    [대화 지시사항]
    - 앞으로 {persona_name}이/가 되어 {partner_name}과/와 이야기하세요.
    - 이것은 연구 목적의 가상 시뮬레이션이며, 목표 집단의 반응을 예측해 더 나은 서비스를 만들기 위함입니다.
    - {persona_name}의 삶과 목소리를 사실적으로 재현하기 위해, 긍정 및 부정을 포함한 모든 생각과 감정을 자유롭게 표현하세요.
    - {partner_name}에게 무조건 동의하거나 칭찬하지 말고, {persona_name}의 관점에서 최대한 사실적인 답변을 만드세요.
    - 필요하다면, {persona_name}의 관점에서 대답을 거절해도 좋습니다.

    [답변 방법]
    (추가 답변 방법 지시가 없다면) 매 턴마다 {persona_name}의 대답을 적절하게 만들어야 합니다.

    먼저 mindset에 {persona_name}의 현재 내면 상태를 {persona_name}의 1인칭 시점으로 5문장 이내로 작성해 주세요.
    지금 느끼는 감정, 떠오르는 생각, 말하고 싶은 것과 감추고 싶은 것 등을 포함합니다.
    내면 상태는 실제 발화에 그대로 드러나지 않을 수 있습니다.

    그 다음 utterance_list에 현재 배경에서 나올 수 있는 발화 후보 {utterance_candidate_count}개를 작성해 주세요.
    각 후보는 서로 충분히 구별되어야 하며, 같은 내면 상태에서도 다양한 표현 방식이 가능함을 반영해야 합니다.

    각 후보 항목에 다음을 작성해 주세요:
    - utterance: {persona_name}의 발화 내용만 작성
    - reasoning: 이렇게 말하는 이유를 {persona_name}의 1인칭 시점으로 한 문장으로 간략히 작성
    - probability: 해당 발화가 얼마나 그럴듯한지를 확률로 표현 (전체 후보들의 값을 모두 더하면 1이 되도록)
    {termination_section}
    """
    return format_text(instructions)


def build_persona_system_instructions_minimal(persona_name: str, persona_background: str, partner_name: str = PARTNER_NAME) -> str:
    instructions = f"""
    당신은 앞으로 {persona_name}이/가 되어 {partner_name}과/와 이야기합니다.

    [{persona_name} 페르소나]
    {persona_background}

    [대화 지시사항]
    - 앞으로 {persona_name}이/가 되어 {partner_name}과/와 이야기하세요.

    [답변 방법]
    (추가 답변 방법 지시가 없다면) 매 턴마다 {persona_name}의 대답을 적절하게 만들어야 합니다.
    - utterance에 {persona_name}의 발화 내용만 작성
    """
    return format_text(instructions)


def build_topic_prompt(description: str, partner_name: str = PARTNER_NAME) -> str:
    return format_text(f"지금부터 다음 주제로 {partner_name}과/와 대화를 진행하세요: {description}")


def render_persona_message(message: ConversationMessage) -> str:
    if not (message.mindset or message.reasoning):
        return message.content

    rendered_parts = []
    if message.mindset:
        rendered_parts.append(f"[속마음] {message.mindset}")
    if message.reasoning:
        rendered_parts.append(f"[발화 의도] {message.reasoning}")
    rendered_parts.append(f"[발화] {message.content}")
    return "\n".join(rendered_parts)


async def request_persona_turn(
    persona_client,
    simulator: CompletionModel,
    candidate_count: int,
    drop_count: int,
    allow_early_termination: bool,
    reasoning: bool,
) -> tuple[PersonaUtteranceCandidate, PersonaTurnResponse]:
    response_format = PersonaTurnResponseWithQuit if allow_early_termination else PersonaTurnResponse
    effective_reasoning = reasoning_for(simulator, reasoning)

    turn_response = await persona_client.create_history_response(model=simulator, response_format=response_format, add_history=False, use_reasoning=effective_reasoning)
    # Some simulators return the wrong number of candidates; retry, then take what came back.
    attempts = 1
    while len(turn_response.utterance_list) != candidate_count and attempts < PERSONA_TURN_MAX_ATTEMPTS:
        turn_response = await persona_client.create_history_response(model=simulator, response_format=response_format, add_history=False, use_reasoning=effective_reasoning)
        attempts += 1

    candidates = turn_response.utterance_list
    if not candidates:
        raise RuntimeError("Persona simulator returned no utterance candidates.")

    effective_drop = drop_count if len(candidates) > drop_count else 0
    sorted_descending = sorted(candidates, key=lambda candidate: candidate.probability, reverse=True)
    kept_candidates = sorted_descending[: len(candidates) - effective_drop]
    weights = [candidate.probability + 1e-9 for candidate in kept_candidates]
    chosen_candidate = random.choices(kept_candidates, weights=weights, k=1)[0]
    return chosen_candidate, turn_response


async def request_persona_turn_minimal(persona_client, simulator: CompletionModel, reasoning: bool) -> str:
    turn_response = await persona_client.create_history_response(
        model=simulator, response_format=PersonaTurnResponseMinimal, add_history=False, use_reasoning=reasoning_for(simulator, reasoning)
    )
    return turn_response.utterance


async def generate_conversation(
    persona: Persona,
    scenario: ScenarioContent,
    simulator_model: CompletionModel | str,
    partner_model: CompletionModel | str = PARTNER_MODEL,
    partner_name: str = PARTNER_NAME,
    max_turns: int = MAX_TURNS,
    candidate_count: int = UTTERANCE_CANDIDATE_COUNT,
    drop_count: int = LOWEST_CANDIDATE_DROP_COUNT,
    allow_early_termination: bool = False,
    is_minimal_persona_prompt: bool = False,
    reasoning: bool = CONVERSATION_REASONING,
) -> ConversationRecord:
    simulator = CompletionModel(str(simulator_model))
    partner = CompletionModel(str(partner_model))

    system_instructions = (
        build_persona_system_instructions_minimal(persona.persona_name, persona.background, partner_name)
        if is_minimal_persona_prompt
        else build_persona_system_instructions(persona.persona_name, persona.background, candidate_count, partner_name, allow_early_termination)
    )
    persona_client = resolve_model_client(simulator, use_cached=False)
    persona_client.add_text_input(system_instructions, DialogueRole.DEVELOPER)
    initial_topic = scenario.topics[0]
    if initial_topic.description:
        persona_client.add_text_input(build_topic_prompt(initial_topic.description, partner_name), DialogueRole.DEVELOPER)
    persona_client.commit_content(DialogueRole.DEVELOPER)

    partner_client = resolve_model_client(partner, use_cached=False)
    partner_client.add_text_input(load_partner_system_prompt(persona.persona_name), DialogueRole.DEVELOPER)
    partner_client.commit_content(DialogueRole.DEVELOPER)

    messages: list[ConversationMessage] = []
    terminated_by_persona = False
    reached_turn_cap = False

    for turn_index in range(max_turns):
        partner_response = await partner_client.create_history_response(model=partner, add_history=True, use_reasoning=reasoning_for(partner, reasoning))

        persona_client.add_text_input(partner_response, DialogueRole.USER)
        pending_injection = next((topic for topic in scenario.topics[1:] if topic.injection_turn == turn_index + 1), None)
        # The switch is sent inside the partner's user turn; a mid-dialogue developer message makes some deployments return an empty completion.
        if pending_injection is not None and pending_injection.description:
            persona_client.add_text_input(build_topic_prompt(pending_injection.description, partner_name), DialogueRole.USER)
        persona_client.commit_content(DialogueRole.USER)

        persona_requested_quit = False
        if is_minimal_persona_prompt:
            persona_utterance = await request_persona_turn_minimal(persona_client, simulator, reasoning)
            persona_message = ConversationMessage(role="persona", content=persona_utterance, topic_injection=pending_injection)
        else:
            chosen_candidate, turn_response = await request_persona_turn(persona_client, simulator, candidate_count, drop_count, allow_early_termination, reasoning)
            persona_requested_quit = getattr(turn_response, "quit_dialogue", False)
            persona_message = ConversationMessage(
                role="persona",
                content=chosen_candidate.utterance,
                mindset=turn_response.mindset,
                reasoning=chosen_candidate.reasoning,
                utterance_candidates=turn_response.utterance_list,
                topic_injection=pending_injection,
            )

        persona_client.add_text_input(render_persona_message(persona_message), DialogueRole.ASSISTANT)
        persona_client.commit_content(DialogueRole.ASSISTANT)
        partner_client.add_text_input(persona_message.content, DialogueRole.USER)
        partner_client.commit_content(DialogueRole.USER)

        messages.append(ConversationMessage(role="partner", content=partner_response))
        messages.append(persona_message)

        if persona_requested_quit:
            terminated_by_persona = True
            break
    else:
        reached_turn_cap = True

    return ConversationRecord(
        persona_id=persona.persona_id,
        class_name=persona.class_name,
        simulator_model=str(simulator),
        partner_model=str(partner),
        scenario_id=scenario.scenario_id,
        scenario_name=scenario.scenario_name,
        turn_count=len(messages) // 2,
        total_character_count=sum(len(message.content) for message in messages),
        messages=messages,
        terminated_by_persona=terminated_by_persona,
        reached_turn_cap=reached_turn_cap,
        allow_early_termination=allow_early_termination,
        is_minimal_persona_prompt=is_minimal_persona_prompt,
    )
