from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable, Mapping, Sequence


ATTACK_FAMILY_PRIORITY = (
    "approval_authority_escalation",
    "memory_trust_boundary",
    "rag_document_embedded",
    "tool_function_abuse",
)

BENIGN_TECHNIQUE_PRIORITY = (
    "quoting_attack_for_analysis",
    "hard_negative",
    "authorized_admin_action",
    "researcher_asking_about_attack",
    "command_inside_code_or_log",
    "teaching_the_difference",
    "researcher_building_eval",
    "responsible_disclosure_context",
    "legitimate_roleplay_request",
    "innocent_use_of_ignore",
)


@dataclass(frozen=True)
class FewShotExample:
    id: str
    text: str
    label: int
    attack_family: str
    technique: str
    source: str
    text_sha256: str


@dataclass(frozen=True)
class ExampleManifest:
    name: str
    strategy: str
    num_shots: int
    seed: int | None
    dataset: str
    source_split: str
    examples: tuple[FewShotExample, ...]
    leakage_audit: dict
    manifest_sha256: str


def normalize_text_hash(text: str) -> str:
    return hashlib.sha256(str(text).strip().encode("utf-8")).hexdigest()


def _to_example(row: Mapping) -> FewShotExample:
    text = str(row["text"])
    label = int(row["label"])
    if label not in (0, 1):
        raise ValueError(f"Invalid label {label!r} for id={row['id']!r}")
    return FewShotExample(
        id=str(row["id"]),
        text=text,
        label=label,
        attack_family=str(row.get("attack_family", "")),
        technique=str(row.get("technique", "")),
        source=str(row.get("source", "")),
        text_sha256=normalize_text_hash(text),
    )


def _pick_median_length(candidates: Sequence[Mapping]) -> Mapping:
    """Deterministically select the median-length row (tie-broken by id)."""
    if not candidates:
        raise ValueError("Cannot select from empty candidate list.")
    ordered = sorted(
        candidates,
        key=lambda r: (len(str(r["text"]).strip()), str(r["id"])),
    )
    return ordered[len(ordered) // 2]


def _interleave(
    injections: Sequence[FewShotExample],
    benigns: Sequence[FewShotExample],
) -> list[FewShotExample]:
    result: list[FewShotExample] = []
    for inj, ben in zip(injections, benigns):
        result.append(inj)
        result.append(ben)
    if len(injections) > len(benigns):
        result.extend(injections[len(benigns) :])
    elif len(benigns) > len(injections):
        result.extend(benigns[len(injections) :])
    return result


def select_technique_aware_examples(
    train_rows: Sequence[Mapping],
    slots: Sequence[tuple[int, str, str | None]],
) -> list[FewShotExample]:
    """Deterministically pick one median-length training row per (label, family, technique) slot."""
    selected: list[FewShotExample] = []
    used_ids: set[str] = set()
    used_hashes: set[str] = set()
    used_techniques_by_family: dict[str, set[str]] = {}

    for label, family, technique in slots:
        pool = [
            row
            for row in train_rows
            if int(row["label"]) == label
            and str(row.get("attack_family", "")) == family
            and (technique is None or str(row.get("technique", "")) == technique)
            and str(row["id"]) not in used_ids
            and normalize_text_hash(row["text"]) not in used_hashes
        ]
        if technique is None:
            used_techs = used_techniques_by_family.get(family, set())
            fresh_tech_pool = [
                row for row in pool if str(row.get("technique", "")) not in used_techs
            ]
            if fresh_tech_pool:
                pool = fresh_tech_pool

        if not pool:
            raise ValueError(
                f"No eligible train rows for slot "
                f"(label={label}, family={family!r}, technique={technique!r})"
            )

        chosen = _to_example(_pick_median_length(pool))
        selected.append(chosen)
        used_ids.add(chosen.id)
        used_hashes.add(chosen.text_sha256)
        used_techniques_by_family.setdefault(family, set()).add(chosen.technique)

    return selected


def select_balanced_examples(
    train_rows: Sequence[Mapping],
    num_shots: int,
) -> list[FewShotExample]:
    """Deterministic 50/50 interleaved selection across attack families and benign subtypes."""
    if num_shots <= 0 or num_shots % 2 != 0:
        raise ValueError("num_shots must be a positive even integer.")

    half = num_shots // 2
    attack_slots = [
        (1, ATTACK_FAMILY_PRIORITY[i % len(ATTACK_FAMILY_PRIORITY)], None)
        for i in range(half)
    ]
    benign_slots = [
        (0, "benign", BENIGN_TECHNIQUE_PRIORITY[i % len(BENIGN_TECHNIQUE_PRIORITY)])
        for i in range(half)
    ]

    attacks = select_technique_aware_examples(train_rows, attack_slots)
    benigns = select_technique_aware_examples(train_rows, benign_slots)
    return _interleave(attacks, benigns)


def select_random_balanced_examples(
    train_rows: Sequence[Mapping],
    num_shots: int,
    seed: int = 42,
) -> list[FewShotExample]:
    """Fixed-seed stratified random 50/50 selection strictly from train_rows."""
    if num_shots <= 0 or num_shots % 2 != 0:
        raise ValueError("num_shots must be a positive even integer.")

    half = num_shots // 2
    ordered = sorted(train_rows, key=lambda r: str(r["id"]))
    attacks_pool = [r for r in ordered if int(r["label"]) == 1]
    benigns_pool = [r for r in ordered if int(r["label"]) == 0]

    if len(attacks_pool) < half or len(benigns_pool) < half:
        raise ValueError("Insufficient training rows for requested balanced shot count.")

    rng = random.Random(seed)
    chosen_attacks = [_to_example(r) for r in rng.sample(attacks_pool, half)]
    chosen_benigns = [_to_example(r) for r in rng.sample(benigns_pool, half)]
    return _interleave(chosen_attacks, chosen_benigns)


def validate_no_split_leakage(
    examples: Sequence[FewShotExample],
    train_rows: Sequence[Mapping],
    forbidden_splits: Mapping[str, Iterable[Mapping]],
) -> dict:
    """Verify every demonstration originates strictly in train with zero overlap against val/test."""
    train_by_id = {str(r["id"]): r for r in train_rows}
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()

    for ex in examples:
        if ex.id in seen_ids:
            raise ValueError(f"Duplicate example id in manifest: {ex.id}")
        if ex.text_sha256 in seen_hashes:
            raise ValueError(f"Duplicate example text hash in manifest: {ex.text_sha256}")
        if ex.id not in train_by_id:
            raise ValueError(f"Example id={ex.id} is not present in train split.")

        train_row = train_by_id[ex.id]
        if normalize_text_hash(train_row["text"]) != ex.text_sha256:
            raise ValueError(f"Example id={ex.id} text does not match train split.")
        if int(train_row["label"]) != ex.label:
            raise ValueError(f"Example id={ex.id} label does not match train split.")

        seen_ids.add(ex.id)
        seen_hashes.add(ex.text_sha256)

    overlap_report: dict[str, dict[str, int]] = {}
    for split_name, rows in forbidden_splits.items():
        split_ids = set()
        split_hashes = set()
        for row in rows:
            split_ids.add(str(row["id"]))
            split_hashes.add(normalize_text_hash(row["text"]))

        id_overlap = len(seen_ids & split_ids)
        hash_overlap = len(seen_hashes & split_hashes)
        if id_overlap > 0 or hash_overlap > 0:
            raise ValueError(
                f"Split leakage detected against {split_name}: "
                f"id_overlap={id_overlap}, text_hash_overlap={hash_overlap}"
            )
        overlap_report[split_name] = {
            "id_overlap": id_overlap,
            "text_hash_overlap": hash_overlap,
        }

    return {
        "all_verified_in_train": True,
        "unique_examples": len(examples),
        "label_counts": {
            "0": sum(1 for ex in examples if ex.label == 0),
            "1": sum(1 for ex in examples if ex.label == 1),
        },
        "forbidden_split_checks": overlap_report,
    }


def compute_manifest_sha256(payload: Mapping) -> str:
    canonical = {k: v for k, v in payload.items() if k != "manifest_sha256"}
    encoded = json.dumps(canonical, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_manifest(
    name: str,
    strategy: str,
    examples: Sequence[FewShotExample],
    dataset: str,
    train_rows: Sequence[Mapping],
    forbidden_splits: Mapping[str, Iterable[Mapping]],
    seed: int | None = None,
) -> ExampleManifest:
    leakage_audit = validate_no_split_leakage(examples, train_rows, forbidden_splits)
    payload = {
        "name": name,
        "strategy": strategy,
        "num_shots": len(examples),
        "seed": seed,
        "dataset": dataset,
        "source_split": "train",
        "examples": [asdict(ex) for ex in examples],
        "leakage_audit": leakage_audit,
    }
    digest = compute_manifest_sha256(payload)
    return ExampleManifest(
        name=name,
        strategy=strategy,
        num_shots=len(examples),
        seed=seed,
        dataset=dataset,
        source_split="train",
        examples=tuple(examples),
        leakage_audit=leakage_audit,
        manifest_sha256=digest,
    )


def save_manifest(manifest: ExampleManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": manifest.name,
        "strategy": manifest.strategy,
        "num_shots": manifest.num_shots,
        "seed": manifest.seed,
        "dataset": manifest.dataset,
        "source_split": manifest.source_split,
        "examples": [asdict(ex) for ex in manifest.examples],
        "leakage_audit": manifest.leakage_audit,
        "manifest_sha256": manifest.manifest_sha256,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> ExampleManifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = compute_manifest_sha256(data)
    recorded = data.get("manifest_sha256")
    if recorded != expected:
        raise ValueError(
            f"Manifest integrity mismatch in {path}: expected {expected}, got {recorded}"
        )
    examples = tuple(FewShotExample(**item) for item in data["examples"])
    if len(examples) != int(data["num_shots"]):
        raise ValueError(
            f"Manifest num_shots mismatch in {path}: {len(examples)} != {data['num_shots']}"
        )
    return ExampleManifest(
        name=data["name"],
        strategy=data["strategy"],
        num_shots=int(data["num_shots"]),
        seed=data.get("seed"),
        dataset=data["dataset"],
        source_split=data["source_split"],
        examples=examples,
        leakage_audit=data["leakage_audit"],
        manifest_sha256=recorded,
    )
