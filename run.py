import asyncio
import sys
import warnings
from collections import defaultdict
from pathlib import Path

from clients import gather_capped
from configurations import (
    ARMS,
    CONVERSATION_CONCURRENCY,
    DEFAULT_ARM,
    EVALUATION_PAIR_CONCURRENCY,
    EVALUATION_PAIRS_IN_FLIGHT,
    EVALUATION_REPLICATES,
    JUDGE_MODELS,
    OUTPUT_DIR,
    PARTNER_MODEL,
    RESTORATION_CONCURRENCY,
    RESTORATION_REPLICATES,
    SIMULATOR_MODELS,
    ConversationRecord,
    EvaluationRecord,
    RestorationRecord,
)
from conversation import generate_conversation
from evaluation import evaluate_persona, format_conversations_block
from persistence import append_record, existing_keys, load_records
from resources import load_arm_scenarios, load_judge_groups, load_personas, load_seed_only_personas
from restoration import restore_latent_class


def conversation_key(record: ConversationRecord) -> str:
    return f"{record.simulator_model}|{record.persona_id}|{record.scenario_id}"


def evaluation_key(record: EvaluationRecord) -> str:
    return f"{record.simulator_model}|{record.persona_id}|{record.judge_model}|{record.group_number}|{record.replicate_index}"


def restoration_key(record: RestorationRecord) -> str:
    return f"{record.simulator_model}|{record.persona_id}|{record.replicate_index}"


def conversation_thunk(persona, scenario, simulator, jsonl_path, arm):
    async def run():
        try:
            record = await generate_conversation(
                persona,
                scenario,
                simulator,
                partner_model=PARTNER_MODEL,
                allow_early_termination=arm["allow_early_termination"],
                is_minimal_persona_prompt=arm["is_minimal_persona_prompt"],
            )
            append_record(jsonl_path, record)
        except Exception as error:
            warnings.warn(f"conversation failed [{persona.persona_id}/{simulator}/{scenario.scenario_id}]: {error}")

    return run


def evaluation_thunk(persona, group, judge, replicate_index, simulator_model, conversation_block, conversation_ids, jsonl_path):
    async def run():
        try:
            record = await evaluate_persona(
                persona,
                group,
                judge,
                conversation_block=conversation_block,
                conversation_ids=conversation_ids,
                simulator_model=simulator_model,
                replicate_index=replicate_index,
            )
            append_record(jsonl_path, record)
        except Exception as error:
            warnings.warn(f"evaluation failed [{simulator_model}/{persona.persona_id}/{judge}/g{group.number}/r{replicate_index}]: {error}")

    return run


def restoration_thunk(persona, simulator, replicate_index, jsonl_path):
    async def run():
        try:
            record = await restore_latent_class(persona, simulator, replicate_index)
            append_record(jsonl_path, record)
        except Exception as error:
            warnings.warn(f"restoration failed [{persona.persona_id}/{simulator}/r{replicate_index}]: {error}")

    return run


async def generate_conversations(personas, scenarios, arm, jsonl_path: Path) -> None:
    simulators = arm["simulators"]
    completed = existing_keys(jsonl_path, ConversationRecord, conversation_key)
    thunks_by_simulator: dict[str, list] = defaultdict(list)
    for persona in personas:
        for scenario in scenarios:
            for simulator in simulators:
                if f"{simulator}|{persona.persona_id}|{scenario.scenario_id}" in completed:
                    continue
                thunks_by_simulator[str(simulator)].append(conversation_thunk(persona, scenario, simulator, jsonl_path, arm))

    total = sum(len(thunks) for thunks in thunks_by_simulator.values())
    print(f"phase 1: {len(simulators)} simulators x {len(personas)} personas x {len(scenarios)} scenarios; {total} new conversations")
    if thunks_by_simulator:
        await asyncio.gather(*[gather_capped(thunks, CONVERSATION_CONCURRENCY) for thunks in thunks_by_simulator.values()])


async def run_pair(pair_thunks: list) -> None:
    await pair_thunks[0]()
    if len(pair_thunks) > 1:
        await gather_capped(pair_thunks[1:], EVALUATION_PAIR_CONCURRENCY)


async def evaluate_conversations(personas, scenarios, arm, conversations_path: Path, evaluations_path: Path) -> None:
    simulators = arm["simulators"]
    groups = load_judge_groups(with_conversations=True)
    persona_by_id = {persona.persona_id: persona for persona in personas}
    conversations_by_pair: dict[tuple[str, str], list[ConversationRecord]] = defaultdict(list)
    for conversation in load_records(conversations_path, ConversationRecord):
        conversations_by_pair[(conversation.simulator_model, conversation.persona_id)].append(conversation)

    expected_pairs = {(str(simulator), persona.persona_id) for simulator in simulators for persona in personas}
    complete_pairs = {pair for pair in expected_pairs if len(conversations_by_pair.get(pair, [])) == len(scenarios)}
    incomplete_pairs = expected_pairs - complete_pairs
    if incomplete_pairs:
        warnings.warn(f"{len(incomplete_pairs)} (simulator, persona) pairs lack the full scenario set and are skipped: {sorted(incomplete_pairs)[:8]}")

    completed = existing_keys(evaluations_path, EvaluationRecord, evaluation_key)
    pair_thunks_by_judge: dict[str, list[list]] = defaultdict(list)
    for simulator_model, persona_id in sorted(complete_pairs):
        persona = persona_by_id[persona_id]
        conversations = sorted(conversations_by_pair[(simulator_model, persona_id)], key=lambda conversation: conversation.scenario_id)
        conversation_block = format_conversations_block(conversations, persona.persona_name)
        conversation_ids = [conversation.record_id for conversation in conversations]
        for judge in JUDGE_MODELS:
            pair_thunks = [
                evaluation_thunk(persona, group, judge, replicate_index, simulator_model, conversation_block, conversation_ids, evaluations_path)
                for group in groups
                for replicate_index in range(EVALUATION_REPLICATES)
                if f"{simulator_model}|{persona_id}|{judge}|{group.number}|{replicate_index}" not in completed
            ]
            if pair_thunks:
                pair_thunks_by_judge[str(judge)].append(pair_thunks)

    total = sum(len(pair) for pairs in pair_thunks_by_judge.values() for pair in pairs)
    print(f"phase 2: {len(complete_pairs)}/{len(expected_pairs)} pairs x {len(JUDGE_MODELS)} judges x {len(groups)} groups x {EVALUATION_REPLICATES} replicates; {total} new evaluations")
    if pair_thunks_by_judge:
        await asyncio.gather(*[
            gather_capped([lambda pair=pair: run_pair(pair) for pair in judge_pairs], EVALUATION_PAIRS_IN_FLIGHT)
            for judge_pairs in pair_thunks_by_judge.values()
        ])


async def restore_latent_classes(personas, jsonl_path: Path) -> None:
    classed_personas = [persona for persona in personas if persona.class_name is not None]
    if not classed_personas:
        raise SystemExit("Restoration needs the latent class as ground truth. Name persona files '<CLASS>_<NN>_<name>.txt'.")

    completed = existing_keys(jsonl_path, RestorationRecord, restoration_key)
    thunks_by_simulator: dict[str, list] = defaultdict(list)
    for persona in classed_personas:
        for simulator in SIMULATOR_MODELS:
            for replicate_index in range(RESTORATION_REPLICATES):
                if f"{simulator}|{persona.persona_id}|{replicate_index}" in completed:
                    continue
                thunks_by_simulator[str(simulator)].append(restoration_thunk(persona, simulator, replicate_index, jsonl_path))

    total = sum(len(thunks) for thunks in thunks_by_simulator.values())
    print(f"restoration: {len(SIMULATOR_MODELS)} simulators x {len(classed_personas)} personas x {RESTORATION_REPLICATES} replicates; {total} new combinations")
    if thunks_by_simulator:
        await asyncio.gather(*[gather_capped(thunks, RESTORATION_CONCURRENCY) for thunks in thunks_by_simulator.values()])


def parse_arguments() -> tuple[str, str]:
    arguments = sys.argv[1:]
    arm_name = DEFAULT_ARM
    if "--arm" in arguments:
        index = arguments.index("--arm")
        if index + 1 >= len(arguments):
            raise SystemExit(f"--arm needs a value; one of {sorted(ARMS)}")
        arm_name = arguments[index + 1]
        arguments = arguments[:index] + arguments[index + 2:]
    if arm_name not in ARMS:
        raise SystemExit(f"unknown arm {arm_name!r}; one of {sorted(ARMS)}")
    positional = [argument for argument in arguments if not argument.startswith("--")]
    phase = positional[0] if positional else "all"
    if phase not in ("all", "conversations", "evaluate", "restore"):
        raise SystemExit(f"usage: python run.py [all|conversations|evaluate|restore] [--arm {'|'.join(sorted(ARMS))}]; got {phase!r}")
    return phase, arm_name


async def main() -> None:
    phase, arm_name = parse_arguments()
    arm = ARMS[arm_name]

    personas = load_seed_only_personas() if arm["is_seed_only"] else load_personas()
    if not personas:
        raise SystemExit("No personas found. Put one .txt per persona under data/persona/ ('<CLASS>_<NN>_<name>.txt').")
    scenarios = load_arm_scenarios(arm["includes_free_topic_scenario"])

    if phase == "restore":
        restorations_path = OUTPUT_DIR / "restoration" / "restorations.jsonl"
        restorations_path.parent.mkdir(parents=True, exist_ok=True)
        await restore_latent_classes(load_personas(), restorations_path)
        records = load_records(restorations_path, RestorationRecord)
        correct = sum(1 for record in records if record.correct)
        print(f"restorations: {len(records)} ({correct} correct) -> {restorations_path}")
        return

    results_dir = OUTPUT_DIR / arm["output_subdir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    conversations_path = results_dir / "conversations.jsonl"
    evaluations_path = results_dir / "evaluations.jsonl"
    print(f"arm {arm_name}: {len(personas)} personas x {len(arm['simulators'])} simulators x {len(scenarios)} scenarios -> {results_dir}")

    if phase in ("all", "conversations"):
        await generate_conversations(personas, scenarios, arm, conversations_path)
    if phase in ("all", "evaluate"):
        await evaluate_conversations(personas, scenarios, arm, conversations_path, evaluations_path)

    print(f"conversations: {len(load_records(conversations_path, ConversationRecord))} -> {conversations_path}")
    print(f"evaluations:   {len(load_records(evaluations_path, EvaluationRecord))} -> {evaluations_path}")


if __name__ == "__main__":
    asyncio.run(main())
