from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

RecordType = TypeVar("RecordType", bound=BaseModel)


def append_record(jsonl_path: Path, record: BaseModel) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json() + "\n")


def load_records(jsonl_path: Path, model_type: type[RecordType]) -> list[RecordType]:
    if not jsonl_path.exists():
        return []
    records: list[RecordType] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(model_type.model_validate_json(line))
    return records


def existing_keys(jsonl_path: Path, model_type: type[RecordType], key_fn: Callable[[RecordType], str]) -> set[str]:
    return {key_fn(record) for record in load_records(jsonl_path, model_type)}
