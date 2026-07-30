"""Feature extraction. One path, imported by training and by serving alike."""

from kanal.features.text import ALLOWED_FIELDS, FORBIDDEN_FIELDS, Example, to_text

__all__ = ["ALLOWED_FIELDS", "FORBIDDEN_FIELDS", "Example", "to_text"]
