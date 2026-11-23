import datetime as dt
import re
from uuid import uuid4

from .configurations import TIMEZONE


########## Common ##########


def now() -> dt.datetime:
    return dt.datetime.now(tz=TIMEZONE)


def uuid() -> str:
    return str(uuid4())


########## Formatting ##########


_JOSA_PAIRS = [
    ("이", "가"),
    ("은", "는"),
    ("을", "를"),
    ("과", "와"),
    ("아", "야"),
    ("으로", "로"),
]

# All marker spellings, longest first so that longer markers ("으로/로") are never partially matched by shorter ones.
_JOSA_MARKER_VARIANTS = sorted(
    [f"{batchim_form}/{vowel_form}" for batchim_form, vowel_form in _JOSA_PAIRS] + [f"{vowel_form}/{batchim_form}" for batchim_form, vowel_form in _JOSA_PAIRS],
    key=len,
    reverse=True,
)
_JOSA_MARKER_PATTERN = re.compile(rf"([가-힣a-zA-Z0-9]+)({'|'.join(map(re.escape, _JOSA_MARKER_VARIANTS))})")

# Reverse lookup from a marker spelling to its canonical (batchim form, vowel form) pair.
_JOSA_MARKER_TO_PAIR = {}
for _batchim_form, _vowel_form in _JOSA_PAIRS:
    _JOSA_MARKER_TO_PAIR[f"{_batchim_form}/{_vowel_form}"] = (_batchim_form, _vowel_form)
    _JOSA_MARKER_TO_PAIR[f"{_vowel_form}/{_batchim_form}"] = (_batchim_form, _vowel_form)


def _select_josa(word: str, batchim_form: str, vowel_form: str) -> str | None:
    """
    Select the correct particle form based on whether the word's final letter has a batchim (final consonant).
    :param word: The word preceding the particle.
    :param batchim_form: Particle form used after a final consonant.
    :param vowel_form: Particle form used after a vowel.
    :return: The selected particle form, or None when the final letter is not Hangul (no conversion possible).
    """
    final_character = word[-1]
    if not ("가" <= final_character <= "힣"):
        return None

    coda_index = (ord(final_character) - ord("가")) % 28
    if (batchim_form, vowel_form) == ("으로", "로"):
        # The ㄹ coda (index 8) behaves as if there were no batchim for this particle (e.g. 서울로, not 서울으로).
        return vowel_form if coda_index in (0, 8) else batchim_form
    return batchim_form if coda_index != 0 else vowel_form


def resolve_josa(text: str) -> str:
    """
    Replace slash particle markers (e.g. "이/가", "가/이") with the particle form matching the preceding word.
    Markers following words that end in non-Hangul characters (English, digits) are left unchanged.
    :param text: Text that may contain slash particle markers.
    :return: The given text with the markers resolved.
    """

    def _replace(match: re.Match) -> str:
        word, marker = match.group(1), match.group(2)
        batchim_form, vowel_form = _JOSA_MARKER_TO_PAIR[marker]
        selected_josa = _select_josa(word, batchim_form, vowel_form)
        if selected_josa is None:
            return match.group(0)
        return word + selected_josa

    return _JOSA_MARKER_PATTERN.sub(_replace, text)


def format_text(text: str) -> str:
    """
    Remove unnecessary indentations from the given triple quotes text.
    Korean particle markers are then resolved.
    :param text: Triple quotes text.
    :return: The given text with the indentations removed and particle markers resolved.
    """
    lines = text.strip().split("\n")
    text_processed = "\n".join([line.strip() for line in lines])
    text_processed = re.sub(r"\n{3,}", "\n\n", text_processed).replace("  ", " ").replace("..", ".")
    return resolve_josa(text_processed)


########## Math ##########


def exponential(coefficient: float, base: float, exponent: float) -> float:
    return coefficient * (base**exponent)
