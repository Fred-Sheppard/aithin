"""Ogham letter specifications and the fixed label mapping.

Label indices always come from aithin/model/label_to_letter.json (never from CSV order).
"""
import json
from dataclasses import dataclass

from .paths import LABEL_MAP

# Direction names follow aithin/synthesis/data/ogham.py.
RIGHT, LEFT, DIAGONAL, ACROSS = "right", "left", "diagonal", "across"


@dataclass(frozen=True)
class Letter:
    name: str
    count: int
    direction: str

    @property
    def group(self) -> str:
        return GROUP_OF_DIRECTION[self.direction]


GROUP_OF_DIRECTION = {RIGHT: "B", LEFT: "H", DIAGONAL: "M", ACROSS: "A"}

_SPECS = [
    Letter("N", 5, RIGHT), Letter("S", 4, RIGHT), Letter("F", 3, RIGHT), Letter("L", 2, RIGHT), Letter("B", 1, RIGHT),
    Letter("Q", 5, LEFT), Letter("C", 4, LEFT), Letter("T", 3, LEFT), Letter("D", 2, LEFT), Letter("H", 1, LEFT),
    Letter("R", 5, DIAGONAL), Letter("Z", 4, DIAGONAL), Letter("NG", 3, DIAGONAL), Letter("G", 2, DIAGONAL), Letter("M", 1, DIAGONAL),
    Letter("I", 5, ACROSS), Letter("E", 4, ACROSS), Letter("U", 3, ACROSS), Letter("O", 2, ACROSS), Letter("A", 1, ACROSS),
]
SPEC = {l.name: l for l in _SPECS}


def load_letters() -> list[str]:
    with open(LABEL_MAP) as f:
        mapping = json.load(f)
    letters = [mapping[str(i)] for i in range(len(mapping))]
    assert set(letters) == set(SPEC), "label_to_letter.json disagrees with the Ogham spec"
    return letters


LETTERS = load_letters()
INDEX = {l: i for i, l in enumerate(LETTERS)}
NUM_CLASSES = len(LETTERS)


def group_of(letter: str) -> str:
    return SPEC[letter].group


def count_of(letter: str) -> int:
    return SPEC[letter].count
