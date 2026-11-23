import re

from service.utilities import format_text

from configurations import (
    CHARACTER_LLM_GROUP_NAME,
    CHARACTER_LLM_GROUP_NUMBER,
    CHARACTER_LLM_QUESTIONS_PATH,
    FREE_TOPIC_SCENARIO_ID,
    PARTNER_PROMPT_PATH,
    PERSONA_DIR,
    PPS_LLM_JUDGE_EXCLUDED_GROUPS,
    PPS_QUESTIONS_PATH,
    SCENARIOS_PATH,
    SEED_ONLY_LIVING_ARRANGEMENT,
    Persona,
    QuestionGroup,
    ScenarioContent,
    ScenarioTopic,
)
from lca import sample_all_seeds

_PPS_HEADER_PATTERN = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*\((.+)\)\s*$")
_SCENARIO_HEADER_PATTERN = re.compile(r"^(\d+)\.\s*(.+)$")
_SCENARIO_TOPIC_PATTERN = re.compile(r"^(첫|\d+턴)\s*대화\s*주제\s*:\s*(.+)$")
_JOSA_MARKERS = ["이/가", "가/이", "은/는", "는/은", "을/를", "를/을", "과/와", "와/과", "아/야", "야/아", "으로/로", "로/으로"]
_JOSA_AFTER_SPACE_PATTERN = re.compile(rf"([가-힣A-Za-z0-9]) +(?={'|'.join(map(re.escape, _JOSA_MARKERS))})")
_PERSONA_FILE_PATTERN = re.compile(r"^([A-Z]+)_(\d+)_(.+)$")


def load_personas() -> list[Persona]:
    personas: list[Persona] = []
    for path in sorted(PERSONA_DIR.glob("*.txt")):
        background = path.read_text(encoding="utf-8").strip()
        match = _PERSONA_FILE_PATTERN.match(path.stem)
        if match is None:
            personas.append(Persona(persona_id=path.stem, persona_name=path.stem, background=background))
            continue
        class_name, number, persona_name = match.groups()
        personas.append(
            Persona(
                persona_id=f"{class_name}_{int(number):02d}",
                persona_name=persona_name,
                background=background,
                class_name=class_name,
            )
        )
    return personas


def load_seed_only_personas() -> list[Persona]:
    seed_text_by_id = {sampled.persona_id: sampled.seed_text for sampled in sample_all_seeds()}
    return [
        persona.model_copy(update={"background": f"{SEED_ONLY_LIVING_ARRANGEMENT}\n{seed_text_by_id[persona.persona_id]}"})
        for persona in load_personas()
    ]


def _topic_injection_turn(label: str) -> int:
    return 1 if label == "첫" else int(label.rstrip("턴"))


def load_scenarios() -> list[ScenarioContent]:
    scenarios: list[ScenarioContent] = []
    number: int | None = None
    name = ""
    topics: list[ScenarioTopic] = []
    for raw_line in SCENARIOS_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        topic_match = _SCENARIO_TOPIC_PATTERN.match(line)
        header_match = _SCENARIO_HEADER_PATTERN.match(line)
        if topic_match:
            topics.append(ScenarioTopic(description=topic_match.group(2).strip(), injection_turn=_topic_injection_turn(topic_match.group(1))))
        elif header_match:
            if number is not None and topics:
                scenarios.append(ScenarioContent(scenario_id=f"scenario_{number}", scenario_name=name, topics=topics))
            number = int(header_match.group(1))
            name = header_match.group(2).strip()
            topics = []
    if number is not None and topics:
        scenarios.append(ScenarioContent(scenario_id=f"scenario_{number}", scenario_name=name, topics=topics))
    return scenarios


def load_arm_scenarios(includes_free_topic_scenario: bool) -> list[ScenarioContent]:
    scenarios = load_scenarios()
    if includes_free_topic_scenario:
        return scenarios
    return [scenario for scenario in scenarios if scenario.scenario_id != FREE_TOPIC_SCENARIO_ID]


def load_partner_system_prompt(persona_name: str) -> str:
    substituted = PARTNER_PROMPT_PATH.read_text(encoding="utf-8").replace("<user_name>", persona_name)
    return format_text(_JOSA_AFTER_SPACE_PATTERN.sub(r"\1", substituted))


def load_pps_groups() -> list[QuestionGroup]:
    groups: list[QuestionGroup] = []
    for raw_line in PPS_QUESTIONS_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        header_match = _PPS_HEADER_PATTERN.match(line)
        if header_match:
            groups.append(QuestionGroup(number=int(header_match.group(1)), name_ko=header_match.group(2).strip(), name_en=header_match.group(3).strip(), questions=[]))
        elif line and groups:
            groups[-1].questions.append(line)
    return groups


def load_character_llm_group() -> QuestionGroup:
    questions = [line.strip() for line in CHARACTER_LLM_QUESTIONS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    return QuestionGroup(number=CHARACTER_LLM_GROUP_NUMBER, name_ko=CHARACTER_LLM_GROUP_NAME, name_en=CHARACTER_LLM_GROUP_NAME, questions=questions)


def load_judge_groups(with_conversations: bool = False) -> list[QuestionGroup]:
    groups = [group for group in load_pps_groups() if group.name_ko not in PPS_LLM_JUDGE_EXCLUDED_GROUPS]
    if with_conversations:
        groups.append(load_character_llm_group())
    return groups
