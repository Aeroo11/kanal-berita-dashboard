"""Canonicalisation and text cleaning.

These are the functions the whole store's identity rests on: get
`canonical_url` wrong and either the same article is stored many times, or two
different articles collide onto one key and one is silently lost.
"""

from __future__ import annotations

import pytest

from kanal.ingest.normalize import (
    article_key,
    canonical_url,
    clean_text,
    strip_html,
    title_fingerprint,
)


class TestCanonicalUrl:
    def test_forces_https(self) -> None:
        assert canonical_url("http://antaranews.com/berita/1/x").startswith("https://")

    def test_drops_www_and_lowercases_host(self) -> None:
        assert canonical_url("https://WWW.AntaraNews.com/berita/1/x") == (
            "https://antaranews.com/berita/1/x"
        )

    def test_drops_fragment(self) -> None:
        assert canonical_url("https://a.com/x#section-2") == "https://a.com/x"

    def test_drops_default_ports_but_keeps_others(self) -> None:
        assert canonical_url("https://a.com:443/x") == "https://a.com/x"
        assert canonical_url("http://a.com:80/x") == "https://a.com/x"
        assert canonical_url("https://a.com:8443/x") == "https://a.com:8443/x"

    def test_strips_trailing_slash_but_keeps_root(self) -> None:
        assert canonical_url("https://a.com/x/") == "https://a.com/x"
        assert canonical_url("https://a.com/") == "https://a.com/"

    def test_collapses_duplicate_slashes(self) -> None:
        assert canonical_url("https://a.com//berita///1") == "https://a.com/berita/1"

    def test_preserves_path_case(self) -> None:
        # Some CMSes serve case-sensitive slugs; folding them would merge
        # genuinely different articles onto one key.
        assert canonical_url("https://a.com/Berita/Judul") == "https://a.com/Berita/Judul"

    @pytest.mark.parametrize(
        "param",
        ["utm_source", "utm_medium", "fbclid", "gclid", "ref", "spec", "pk_campaign"],
    )
    def test_drops_tracking_parameters(self, param: str) -> None:
        assert canonical_url(f"https://a.com/x?{param}=rss") == "https://a.com/x"

    def test_keeps_meaningful_parameters(self) -> None:
        # `page` genuinely addresses different content; dropping it would
        # collapse distinct pages onto one key.
        assert canonical_url("https://a.com/x?page=2") == "https://a.com/x?page=2"

    def test_parameter_order_does_not_change_identity(self) -> None:
        assert canonical_url("https://a.com/x?b=2&a=1") == canonical_url("https://a.com/x?a=1&b=2")

    def test_is_idempotent(self) -> None:
        once = canonical_url("http://WWW.a.com//x/?utm_source=rss#frag")
        assert canonical_url(once) == once

    @pytest.mark.parametrize("bad", ["", "   ", "not a url", "ftp://a.com/x", "/relative/path"])
    def test_rejects_unusable_input(self, bad: str) -> None:
        # Returning a mangled string here would create key collisions, which is
        # strictly worse than refusing the row.
        with pytest.raises(ValueError):
            canonical_url(bad)


class TestArticleKey:
    def test_all_real_world_variants_collapse_to_one_key(self) -> None:
        variants = [
            "https://www.antaranews.com/berita/123/judul?utm_source=rss&utm_medium=feed",
            "https://www.antaranews.com/berita/123/judul/",
            "http://antaranews.com/berita/123/judul#top",
            "https://antaranews.com//berita/123/judul",
        ]
        assert len({article_key(v) for v in variants}) == 1

    def test_different_articles_get_different_keys(self) -> None:
        assert article_key("https://a.com/berita/1") != article_key("https://a.com/berita/2")

    def test_is_hex_sha256(self) -> None:
        key = article_key("https://a.com/x")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


class TestStripHtml:
    def test_removes_the_antara_thumbnail(self) -> None:
        raw = (
            '<img align="left" border="0" '
            'src="https://cdn.antaranews.com/cache/800x533/2026/07/16/iran.jpg"/>'
            "Iran menyatakan sikap resmi."
        )
        out = strip_html(raw)
        assert "cdn.antaranews.com" not in out
        assert "img" not in out
        assert "Iran menyatakan sikap resmi." in out

    def test_does_not_fuse_words_across_block_tags(self) -> None:
        assert "satu dua" in strip_html("<p>satu</p><p>dua</p>")

    def test_decodes_entities(self) -> None:
        assert strip_html("Ekonomi &amp; Bisnis") == "Ekonomi & Bisnis"

    def test_passes_plain_text_through_untouched(self) -> None:
        assert strip_html("tidak ada markup di sini") == "tidak ada markup di sini"

    def test_survives_malformed_markup(self) -> None:
        # Unclosed tags and stray brackets appear in real feeds; losing the
        # summary entirely would lose a real article.
        assert "teks" in strip_html("<div><span>teks<<< broken")


class TestCleanText:
    @pytest.mark.parametrize(
        "raw",
        [
            "Liputan6.com, Jakarta - Isi berita.",
            "KOMPAS.com - Isi berita.",
            "CNN Indonesia - Isi berita.",
            "TEMPO.CO, Jakarta - Isi berita.",
            # Republika's real forms: the city varies, sometimes carries a
            # trailing comma, and the dash may be doubled.
            "REPUBLIKA.CO.ID, JAKARTA -- Isi berita.",
            "REPUBLIKA.CO.ID, GIANYAR, – Isi berita.",
            "REPUBLIKA.CO.ID, BANDUNG – Isi berita.",
            "REPUBLIKA.CO.ID, SUKABUMI - Isi berita.",
        ],
    )
    def test_strips_publisher_boilerplate(self, raw: str) -> None:
        # The prefix identifies the publisher, and publishers have different
        # section mixes — so it is a partial label, not decoration.
        out = clean_text(raw)
        assert out == "Isi berita."

    def test_strips_stacked_prefixes(self) -> None:
        assert clean_text("Liputan6.com, Jakarta - Jakarta - Isi berita.") == "Isi berita."

    def test_strips_trailing_cross_promotion(self) -> None:
        assert clean_text("Isi berita. Baca juga: Artikel lain") == "Isi berita."

    def test_collapses_whitespace(self) -> None:
        assert clean_text("  a\n\n b\t c ") == "a b c"

    def test_handles_none_and_empty(self) -> None:
        assert clean_text(None) == ""
        assert clean_text("") == ""

    def test_strips_html_before_boilerplate(self) -> None:
        # Order matters: the prefix is only recognisable once markup is gone.
        raw = '<img src="https://cdn.example.com/x.jpg"/>Liputan6.com, Jakarta - Isi.'
        assert clean_text(raw) == "Isi."

    def test_strips_boilerplate_written_with_html_entities(self) -> None:
        # Republika sends "REPUBLIKA.CO.ID,&nbsp;BANDUNG &ndash; ". The entities
        # decode to a non-breaking space and an en dash, so the pattern only
        # matches after strip_html and whitespace normalisation have run — which
        # is the reason clean_text does them in that order.
        raw = "REPUBLIKA.CO.ID,&nbsp;BANDUNG &ndash;&nbsp;Isi berita."
        assert clean_text(raw) == "Isi berita."

    def test_does_not_strip_a_publisher_name_mid_sentence(self) -> None:
        # The patterns are anchored. A story *about* Republika keeps its text.
        raw = "Menteri menyebut REPUBLIKA.CO.ID sebagai contoh media digital."
        assert clean_text(raw) == raw


class TestTitleFingerprint:
    def test_ignores_punctuation_and_case(self) -> None:
        # A wire story republished elsewhere keeps its headline but picks up
        # punctuation and casing differences.
        a = title_fingerprint("Jokowi Resmikan Bendungan, Warga Bersyukur")
        b = title_fingerprint("jokowi resmikan bendungan warga bersyukur")
        assert a == b

    def test_different_headlines_differ(self) -> None:
        assert title_fingerprint("Harga emas naik") != title_fingerprint("Harga emas turun")
