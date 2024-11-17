from typing import Iterable, Optional
from enum import Enum


class MorphemeType(Enum):
    PREF = "PREF"
    ROOT = "ROOT"
    SUFF = "SUFF"
    END = "END"
    POST = "POST"
    HYPN = "HYPN"
    LINK = "LINK"


class Morpheme(object):
    def __init__(
            self,
            label: MorphemeType,
            text: str
    ):
        self.label = label
        self.text = text


class MorphemeParsing(object):
    def __init__(self, morphemes: Optional[Iterable[Morpheme]]):
        self.morphemes = morphemes
