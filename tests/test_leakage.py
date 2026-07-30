"""The label must not reach the model by any route except the text.

The label *is* feed provenance, and it leaks into the URL: CNN's article URLs
contain their section 100% of the time, Liputan6's 98.6%, Republika's 35.6%,
ANTARA's 4.1%. A model that sees any of that scores near-perfectly and has
learnt nothing about Indonesian.

These tests are the structural defence. The interesting one is
`test_forbidden_fields_are_absent_not_ignored`: `Example` does not carry the URL
at all, so leaking it is a type error at the call site rather than a quietly
better number three weeks later.
"""

from __future__ import annotations

import dataclasses

import pytest

from kanal.features.text import ALLOWED_FIELDS, FORBIDDEN_FIELDS, Example, to_text


class TestTheStructuralDefence:
    def test_forbidden_fields_are_absent_not_ignored(self) -> None:
        """The defence is the type, not the discipline.

        A struct carrying `canonical_url` and documenting "do not use this" is a
        comment. A struct without the field cannot be misused.
        """
        present = {f.name for f in dataclasses.fields(Example)}
        assert present == set(ALLOWED_FIELDS)

        for forbidden in FORBIDDEN_FIELDS:
            assert forbidden not in present, f"{forbidden} must not be reachable from a feature"

    def test_constructing_with_a_forbidden_field_fails(self) -> None:
        with pytest.raises(TypeError):
            Example(title="x", summary="y", source="cnn")  # type: ignore[call-arg]

    def test_the_forbidden_list_covers_every_leaky_column(self) -> None:
        # Named explicitly so adding a new leaky column to the warehouse without
        # adding it here shows up as a review question rather than never.
        for column in (
            "canonical_url",
            "raw_link",
            "source",
            "channel",
            "feed_id",
            "url_leaks_label",
            "cluster_id",
        ):
            assert column in FORBIDDEN_FIELDS


class TestTextIsInvariantToProvenance:
    def test_identical_text_from_different_publishers_is_identical(self) -> None:
        # The property that matters: two outlets running the same headline
        # produce the same model input, so nothing about *which* outlet can be
        # learnt.
        a = Example(title="Harga emas menguat", summary="Perdagangan hari ini positif.")
        b = Example(title="Harga emas menguat", summary="Perdagangan hari ini positif.")
        assert to_text(a) == to_text(b)

    def test_publisher_boilerplate_is_stripped(self) -> None:
        """Leakage arriving by a second route.

        Left in, a model can learn "starts with Liputan6.com" as a proxy for the
        outlet — and the outlet correlates with the label by construction.
        """
        cases = [
            "Liputan6.com, Jakarta - Harga emas menguat tajam.",
            "REPUBLIKA.CO.ID, JAKARTA -- Harga emas menguat tajam.",
            "Jakarta (ANTARA) - Harga emas menguat tajam.",
        ]
        cleaned = {to_text(Example(title="Emas", summary=c)) for c in cases}
        assert len(cleaned) == 1, f"boilerplate survived: {cleaned}"
        assert "Liputan6" not in next(iter(cleaned))
        assert "ANTARA" not in next(iter(cleaned))

    def test_non_breaking_spaces_normalise_to_ordinary_ones(self) -> None:
        """Typography is a route for provenance to leak.

        Publishers differ on which invisible characters they emit. Without NFKC
        the same headline from two outlets tokenises differently, and that
        difference correlates with the outlet \u2014 which correlates with the
        label.

        Written with an explicit escape rather than a literal non-breaking
        space, so the thing under test is visible in the source instead of being
        a character that looks exactly like an ordinary one.
        """
        ordinary = Example(title="Menteri menolak usulan itu")
        with_nbsp = Example(title="Menteri\u00a0menolak\u00a0usulan itu")
        assert to_text(with_nbsp) == to_text(ordinary)

    def test_full_width_characters_fold_to_ascii(self) -> None:
        # The other common NFKC case. A publisher emitting full-width digits
        # would otherwise produce tokens no other publisher can match.
        assert to_text(Example(title="Rp\uff11\uff10\uff10 ribu")) == to_text(
            Example(title="Rp100 ribu")
        )

    def test_curly_quotes_are_preserved_but_consistently(self) -> None:
        # NFKC deliberately leaves these alone: a curly quote is a meaningful
        # typographic choice, not an encoding artefact. That is fine \u2014 what
        # the leakage defence needs is determinism, not aggressive folding, so
        # that no outlet can be identified by how the function treated its text.
        curly = Example(title="Menteri \u201cmenolak\u201d usulan")
        assert to_text(curly) == to_text(curly)
        assert "\u201c" in to_text(curly)

    def test_control_characters_are_dropped(self) -> None:
        assert to_text(Example(title="Harga\x00 emas\x07 naik")) == "Harga emas naik"

    def test_whitespace_is_collapsed(self) -> None:
        assert to_text(Example(title="  Harga   emas \n\n naik  ")) == "Harga emas naik"


class TestTextComposition:
    def test_title_and_summary_are_joined(self) -> None:
        out = to_text(Example(title="Harga emas naik", summary="Analis memperkirakan tren."))
        assert out == "Harga emas naik. Analis memperkirakan tren."

    def test_a_missing_summary_leaves_only_the_title(self) -> None:
        assert to_text(Example(title="Harga emas naik")) == "Harga emas naik"

    def test_a_missing_title_leaves_only_the_summary(self) -> None:
        assert to_text(Example(title="", summary="Analis memperkirakan tren.")) == (
            "Analis memperkirakan tren."
        )

    def test_a_summary_that_was_only_boilerplate_leaves_the_title(self) -> None:
        assert to_text(Example(title="Harga emas naik", summary="Liputan6.com, Jakarta - ")) == (
            "Harga emas naik"
        )

    def test_is_deterministic(self) -> None:
        # Train and serve call this at different times in different processes.
        # If it were not deterministic, skew would be unavoidable.
        e = Example(title="Timnas menang", summary="Pertandingan berakhir 2-0.")
        assert to_text(e) == to_text(e) == to_text(Example(title=e.title, summary=e.summary))

    def test_casing_is_preserved_for_the_vectoriser_to_decide(self) -> None:
        # Lowercasing here would destroy a signal a transformer could use later.
        # Everything genuinely destructive happens once, in to_text; this does
        # not have to be.
        assert "Harga" in to_text(Example(title="Harga emas naik"))
