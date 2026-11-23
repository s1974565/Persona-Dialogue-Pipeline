"""Latent class analysis over the 11 health-behavior items of the 2020 Korea Elderly Survey.
Source: Cho & Lee (2024), Journal of Affective Disorders 361:172-181 (emission/prior values are the paper's)."""
from enum import StrEnum

import numpy as np
from pydantic import BaseModel

from configurations import (
    ADDITIONAL_SAMPLES_PER_CLASS,
    INITIAL_SAMPLES_PER_CLASS,
    SAMPLING_SEED,
    SEED_DUMP_PATH,
)


class LatentClass(StrEnum):
    CH = "CH"
    MHI = "MHI"
    UA = "UA"


CLASSES: list[str] = [latent_class.value for latent_class in LatentClass]


class SurveyItem(BaseModel):
    """One health-behavior survey item with its binarisation labels and statements.

    healthy_label / unhealthy_label express the original survey response when the healthy
    code is 1 / 0. healthy_statement / unhealthy_statement are concise natural-language
    binarised statements (original-data form, detailed frequencies omitted) for use as seeds.
    """

    key: str
    question_number: str
    section: str
    question: str
    options: str
    healthy_label: str
    unhealthy_label: str
    healthy_statement: str
    unhealthy_statement: str


ITEMS: list[SurveyItem] = [
    SurveyItem(
        key="smoke", question_number="문B7", section="B. 건강상태",
        question="귀하께서는 현재 담배를 피우십니까?",
        options="1) 예(현재 피움)   2) 아니요",
        healthy_label="2) 아니요 (비흡연)",
        unhealthy_label="1) 예 (현재 흡연)",
        healthy_statement="현재 담배를 피우지 않는다",
        unhealthy_statement="현재 담배를 피운다",
    ),
    SurveyItem(
        key="drink", question_number="문B8", section="B. 건강상태",
        question="귀하께서는 지난 1년간 술을 얼마나 자주 마셨습니까?",
        options=("0) 최근 1년간 전혀 마시지 않았다   1) 연 1회 이상~12회 미만   "
                 "2) 한 달에 1회 정도   3) 한 달에 2~3회 정도   4) 일주일에 1회 정도   "
                 "5) 일주일에 2~3회 정도   6) 일주일에 4~6회 정도   7) 매일"),
        healthy_label="0) 최근 1년간 전혀 마시지 않았다 (비음주)",
        unhealthy_label="1)~7) 음주 (지난 1년간 음주; 원자료 빈도분포 없어 세부빈도 미지정)",
        healthy_statement="지난 1년간 술을 전혀 마시지 않았다 (비음주)",
        unhealthy_statement="지난 1년간 술을 마신 적이 있다 (음주)",
    ),
    SurveyItem(
        key="fruit", question_number="문B11-3", section="B. 건강상태(영양관리, 지난 1개월)",
        question="(영양관리) 과일을 거의 먹지 않는다",
        options="1) 그렇다   2) 아니다",
        healthy_label="2) 아니다 (과일 섭취함)",
        unhealthy_label="1) 그렇다 (과일 거의 안 먹음)",
        healthy_statement="지난 1개월간 과일을 챙겨 먹는다",
        unhealthy_statement="지난 1개월간 과일을 거의 먹지 않는다",
    ),
    SurveyItem(
        key="veg", question_number="문B11-3", section="B. 건강상태(영양관리, 지난 1개월)",
        question="(영양관리) 채소를 거의 먹지 않는다",
        options="1) 그렇다   2) 아니다",
        healthy_label="2) 아니다 (채소 섭취함)",
        unhealthy_label="1) 그렇다 (채소 거의 안 먹음)",
        healthy_statement="지난 1개월간 채소를 챙겨 먹는다",
        unhealthy_statement="지난 1개월간 채소를 거의 먹지 않는다",
    ),
    SurveyItem(
        key="dairy", question_number="문B11-3", section="B. 건강상태(영양관리, 지난 1개월)",
        question="(영양관리) 유제품(우유 등)을 거의 먹지 않는다",
        options="1) 그렇다   2) 아니다",
        healthy_label="2) 아니다 (유제품 섭취함)",
        unhealthy_label="1) 그렇다 (유제품 거의 안 먹음)",
        healthy_statement="지난 1개월간 우유 등 유제품을 챙겨 먹는다",
        unhealthy_statement="지난 1개월간 우유 등 유제품을 거의 먹지 않는다",
    ),
    SurveyItem(
        key="exercise", question_number="문B9", section="B. 건강상태",
        question=("귀하께서는 평소 운동을 하십니까?(지속적으로 10분 이상 운동) "
                  "— 일주일 운동 일수(B9-1) × 1회 운동 시간(B9-2) 으로 주당 운동 분 산출 후 150분 기준 이진화"),
        options="문B9 1) 예  2) 아니요   ·  문B9-1 일주일에 [ ]일   ·  문B9-2 1회당 [ ]분",
        healthy_label="주 150분 이상 운동",
        unhealthy_label="주 150분 미만 또는 미운동 (세부 일수/시간 미지정)",
        healthy_statement="평소 운동을 주 150분 이상 한다",
        unhealthy_statement="평소 주 150분 미만으로 운동하거나 운동을 하지 않았다",
    ),
    SurveyItem(
        key="leisure", question_number="문D3", section="D. 여가 및 사회활동(지난 1년간)",
        question=("귀하께서는 지난 1년간 여가문화 활동"
                  "(TV시청 및 라디오 청취, 여행을 제외한 취미활동)을 하셨습니까?"),
        options="1) 예   2) 아니요",
        healthy_label="1) 예 (참여)",
        unhealthy_label="2) 아니요 (미참여)",
        healthy_statement="지난 1년간 여가문화 활동(TV·라디오 시청/청취 및 여행 제외)에 참여한 적이 있다",
        unhealthy_statement="지난 1년간 여가문화 활동(TV·라디오 시청/청취 및 여행 제외)에 참여한 적이 없다",
    ),
    SurveyItem(
        key="social", question_number="문D6", section="D. 여가 및 사회활동(지난 1년간)",
        question=("귀하는 지난 1년간 동호회(클럽활동), 친목 단체(동창회, 계모임 등), "
                  "정치사회 단체 활동에 참여하셨습니까? "
                  "[세 단체 각각: 1) 참여함  2) 참여하지 않았음]"),
        options="동호회 / 친목단체 / 정치사회단체 각각  1) 참여함   2) 참여하지 않았음",
        healthy_label="세 단체 중 1개 이상 참여 (가정)",
        unhealthy_label="세 단체 모두 미참여",
        healthy_statement="지난 1년간 사회단체활동(동호회, 친목모임, 정치사회단체 등)에 한 곳 이상 참여한 적이 있다",
        unhealthy_statement="지난 1년간 사회단체활동(동호회, 친목모임, 정치사회단체 등)에 전혀 참여한 적이 없다",
    ),
    SurveyItem(
        key="education", question_number="문D5", section="D. 여가 및 사회활동(지난 1년간)",
        question=("귀하께서는 지난 1년간 교육(학습활동) "
                  "(TV 및 인터넷 강좌 등 온라인 학습활동 포함)에 참여하신 적이 있으십니까?"),
        options="1) 예   2) 아니요",
        healthy_label="1) 예 (참여)",
        unhealthy_label="2) 아니요 (미참여)",
        healthy_statement="지난 1년간 학습(교육) 활동(TV·인터넷 강좌 등 온라인 학습 포함)에 참여한 적이 있다",
        unhealthy_statement="지난 1년간 학습(교육) 활동(TV·인터넷 강좌 등 온라인 학습 포함)에 참여한 적이 없다",
    ),
    SurveyItem(
        key="checkup", question_number="문B13-1", section="B. 건강상태",
        question="귀하께서는 지난 2년간 건강검진을 받은 적이 있으십니까?(단, 치매검진 제외)",
        options="1) 예   2) 아니요",
        healthy_label="1) 예 (건강검진 받음)",
        unhealthy_label="2) 아니요 (건강검진 안 받음)",
        healthy_statement="지난 2년간 건강검진(치매검진 제외)을 받은 적이 있다",
        unhealthy_statement="지난 2년간 건강검진(치매검진 제외)을 받은 적이 없다",
    ),
    SurveyItem(
        key="dementia", question_number="문B13-2", section="B. 건강상태",
        question="귀하께서는 지난 2년간 치매검진을 받은 적이 있으십니까?",
        options="1) 예   2) 아니요",
        healthy_label="1) 예 (치매검진 받음)",
        unhealthy_label="2) 아니요 (치매검진 안 받음)",
        healthy_statement="지난 2년간 치매검진을 받은 적이 있다",
        unhealthy_statement="지난 2년간 치매검진을 받은 적이 없다",
    ),
]
ITEM_KEYS: list[str] = [item.key for item in ITEMS]
ITEM_BY_KEY: dict[str, SurveyItem] = {item.key: item for item in ITEMS}

# P(healthy=1 | class), from the paper's Fig.1. Rows follow ITEM_KEYS; columns follow CLASSES.
# Note: social MHI=0.0003 is verbatim (4 decimals); checkup MHI=0.00 is rounded.
EMISSION_PROBABILITIES = np.array([
    # CH     MHI      UA
    [0.93,  0.94,   0.85],   # smoke    : don't smoke
    [0.69,  0.70,   0.66],   # drink    : don't drink
    [0.97,  0.87,   0.23],   # fruit    : eat
    [0.99,  0.99,   0.63],   # veg      : eat
    [0.85,  0.77,   0.15],   # dairy    : eat
    [0.38,  0.18,   0.33],   # exercise : >=150 min/week
    [0.82,  0.67,   0.74],   # leisure  : participate
    [0.03,  0.0003, 0.03],   # social   : participate
    [0.11,  0.10,   0.12],   # education: participate
    [0.97,  0.00,   0.83],   # checkup  : yes
    [0.57,  0.02,   0.58],   # dementia : yes
])
assert EMISSION_PROBABILITIES.shape == (len(ITEM_KEYS), len(CLASSES))

# Class prior = modal-assignment proportions (n / 3137), reported by the paper.
CLASS_N: dict[str, int] = {"CH": 2150, "MHI": 744, "UA": 243}
PRIOR = np.array([CLASS_N[class_name] / sum(CLASS_N.values()) for class_name in CLASSES])

# Clamp emission probabilities to [epsilon, 1-epsilon] to avoid degenerate (zero-likelihood) values.
# epsilon=1e-4 only lifts checkup's 0.00 to 1e-4 and leaves social's 0.0003 untouched.
DEFAULT_EPSILON = 1e-4


class SampledPersona(BaseModel):
    persona_id: str
    class_name: LatentClass
    healthy_code: dict[str, int]
    seed_text: str


def sample_healthy_codes(class_name: str, count: int = 1, seed: int | None = None) -> list[dict[str, int]]:
    """
    Sample healthy codes from a class by drawing each item from Bernoulli(P[item, class]).
    :param class_name: Latent class to sample from.
    :param count: Number of codes to sample.
    :param seed: RNG seed for reproducibility.
    :return: One item-key to 0/1 mapping per sampled persona.
    """
    if class_name not in CLASSES:
        raise ValueError(f"class_name must be one of {CLASSES}: {class_name!r}")
    column_index = CLASSES.index(class_name)
    random_generator = np.random.default_rng(seed)
    emission = EMISSION_PROBABILITIES[:, column_index]
    draws = (random_generator.random((count, len(ITEM_KEYS))) < emission).astype(int)
    return [{item_key: int(draws[sample_index, item_index]) for item_index, item_key in enumerate(ITEM_KEYS)} for sample_index in range(count)]


def render_persona_statements(healthy_code: dict[str, int]) -> str:
    """
    Render a healthy code as the bullet list of binarised statements used as the authoring seed.
    :param healthy_code: Item key to 0/1 healthy response.
    :return: The bullet-list statements.
    """
    lines = []
    for item in ITEMS:
        statement = item.healthy_statement if healthy_code[item.key] == 1 else item.unhealthy_statement
        lines.append(f"- {statement}")
    return "\n".join(lines)


def sample_seed_set(per_class: int, seed: int, start_index: int) -> list[SampledPersona]:
    """
    Sample one seed set: ``per_class`` personas from each latent class, numbered from ``start_index``.
    :param per_class: Personas to draw per class.
    :param seed: Base RNG seed; each class is offset by its index.
    :param start_index: First persona number within each class.
    :return: The sampled personas.
    """
    personas: list[SampledPersona] = []
    for class_index, class_name in enumerate(CLASSES):
        for offset, healthy_code in enumerate(sample_healthy_codes(class_name, count=per_class, seed=seed + class_index)):
            personas.append(
                SampledPersona(
                    persona_id=f"{class_name}_{start_index + offset:02d}",
                    class_name=LatentClass(class_name),
                    healthy_code=healthy_code,
                    seed_text=render_persona_statements(healthy_code),
                )
            )
    return personas


def sample_all_seeds() -> list[SampledPersona]:
    initial = sample_seed_set(INITIAL_SAMPLES_PER_CLASS, SAMPLING_SEED, start_index=1)
    additional = sample_seed_set(ADDITIONAL_SAMPLES_PER_CLASS, SAMPLING_SEED + 1, start_index=INITIAL_SAMPLES_PER_CLASS + 1)
    return sorted(initial + additional, key=lambda persona: (CLASSES.index(persona.class_name), int(persona.persona_id.split("_")[1])))


def posterior(healthy_code: dict, prior: np.ndarray | None = None, epsilon: float = DEFAULT_EPSILON) -> dict[str, float]:
    """
    Return P(class | y) over the three classes, computed as P(C=k | y) ∝ π_k · Π_i p_ik^{y_i}
    (1-p_ik)^{1-y_i} in log-space then softmax. Items whose value is None or negative are treated as
    missing (MAR) and dropped from the product.
    :param healthy_code: Item key to 0/1 healthy response (missing keys are dropped).
    :param prior: Class prior; defaults to the paper's modal-assignment proportions.
    :param epsilon: Clamp applied to emission probabilities to avoid zero likelihood.
    :return: Posterior probability per class.
    """
    prior_distribution = PRIOR if prior is None else np.asarray(prior, float)
    prior_distribution = prior_distribution / prior_distribution.sum()
    clamped_emission = np.clip(EMISSION_PROBABILITIES, epsilon, 1.0 - epsilon)
    log_posterior = np.log(prior_distribution).astype(float).copy()
    for item_index, item_key in enumerate(ITEM_KEYS):
        healthy_value = healthy_code.get(item_key, None)
        if healthy_value is None or int(healthy_value) < 0:
            continue
        if int(healthy_value) == 1:
            log_posterior += np.log(clamped_emission[item_index])
        else:
            log_posterior += np.log(1.0 - clamped_emission[item_index])
    weights = np.exp(log_posterior - log_posterior.max())
    normalised = weights / weights.sum()
    return {class_name: float(normalised[index]) for index, class_name in enumerate(CLASSES)}


def posterior_vector(healthy_code: dict, prior: np.ndarray | None = None, epsilon: float = DEFAULT_EPSILON) -> np.ndarray:
    posterior_by_class = posterior(healthy_code, prior=prior, epsilon=epsilon)
    return np.array([posterior_by_class[class_name] for class_name in CLASSES])


def classify(healthy_code: dict, prior: np.ndarray | None = None, epsilon: float = DEFAULT_EPSILON) -> str:
    posterior_by_class = posterior(healthy_code, prior=prior, epsilon=epsilon)
    return max(posterior_by_class, key=posterior_by_class.get)


def cross_entropy(posterior_by_class: dict[str, float], true_class: str, epsilon: float = DEFAULT_EPSILON) -> float:
    """
    Cross-entropy of a one-hot ground truth against the posterior, i.e. -log P(true_class).
    :param posterior_by_class: Posterior probability per class.
    :param true_class: Ground-truth class.
    :param epsilon: Floor applied to the true-class probability.
    :return: The cross-entropy.
    """
    return float(-np.log(max(posterior_by_class[true_class], epsilon)))


def encode_raw_answers(raw: dict) -> dict:
    """
    Convert raw survey responses into a healthy code dict (item_key -> 0/1).

    Expected raw keys (original option numbering; absent keys become missing):
      smoke       : 1=예(흡연), 2=아니요(비흡연)
      drink_freq  : 0~7 (Q B8 option number; 0=non-drinker)
      fruit/veg/dairy : 1=그렇다(거의 안 먹음), 2=아니다(먹음)
      exercise    : 'exercise_ge150'=bool  or  ('exercise_yes'(1/2), 'exercise_days', 'exercise_min')
      leisure/education/checkup/dementia : 1=예, 2=아니요
      social      : [동호회, 친목, 정치사회] each 1=참여/2=미참여  or  bool(any participation)
    :param raw: Raw survey responses keyed as above.
    :return: The healthy code (item_key -> 0/1) for the present items.
    """
    healthy_code: dict = {}
    if "smoke" in raw:
        healthy_code["smoke"] = 1 if int(raw["smoke"]) == 2 else 0
    if "drink_freq" in raw:
        healthy_code["drink"] = 1 if int(raw["drink_freq"]) == 0 else 0
    for item_key in ("fruit", "veg", "dairy"):
        if item_key in raw:
            healthy_code[item_key] = 1 if int(raw[item_key]) == 2 else 0
    if "exercise_ge150" in raw:
        healthy_code["exercise"] = 1 if bool(raw["exercise_ge150"]) else 0
    elif "exercise_yes" in raw:
        if int(raw["exercise_yes"]) == 2:
            healthy_code["exercise"] = 0
        else:
            exercise_days = float(raw.get("exercise_days", 0))
            exercise_minutes = float(raw.get("exercise_min", 0))
            healthy_code["exercise"] = 1 if exercise_days * exercise_minutes >= 150 else 0
    for item_key in ("leisure", "education", "checkup", "dementia"):
        if item_key in raw:
            healthy_code[item_key] = 1 if int(raw[item_key]) == 1 else 0
    if "social" in raw:
        social_value = raw["social"]
        if isinstance(social_value, (list, tuple)):
            healthy_code["social"] = 1 if any(int(group) == 1 for group in social_value) else 0
        else:
            healthy_code["social"] = 1 if bool(social_value) else 0
    return healthy_code


def main() -> None:
    all_personas = sample_all_seeds()
    dump = "\n\n".join(f"### {persona.persona_id} ({persona.class_name})\n{persona.seed_text}" for persona in all_personas)
    SEED_DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEED_DUMP_PATH.write_text(dump, encoding="utf-8")
    print(f"{len(all_personas)} seed conditions -> {SEED_DUMP_PATH}")


if __name__ == "__main__":
    main()
