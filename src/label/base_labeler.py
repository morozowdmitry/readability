from typing import Optional, Set

from src.data_handlers.text import Text
from src.label.labels import LabelType


class AlreadyLabeledException(BaseException):
    pass


class WrongLabelTypeException(BaseException):
    pass


class BaseLabeler(object):
    def __init__(self):
        self.labels: Set[LabelType] = set()

    def run(self, text: Text, labels: Set[LabelType]) -> Text:
        self._validate_labels(text_labels=text.labels, labels=labels)
        self._label(text=text, labels=labels)
        self._update_labels(text=text, added_labels=labels)
        return text

    def _validate_labels(self, text_labels: Set[LabelType], labels: Set[LabelType]):
        if any(x in text_labels for x in labels):
            raise AlreadyLabeledException
        if any(x not in self.labels for x in labels):
            raise WrongLabelTypeException

    def _label(self, text: Text, labels: Set[LabelType]) -> Text:
        return text

    def _update_labels(self, text: Text, added_labels: Set[LabelType]) -> Text:
        text.labels.update(added_labels)
        return text

