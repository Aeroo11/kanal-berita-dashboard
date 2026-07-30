"""MinHash near-duplicate clustering.

The property that matters: the same story written up two ways must land in one
cluster, and unrelated stories must not. Everything downstream splits by cluster,
so a clustering that over-merges silently shrinks the effective dataset, and one
that under-merges lets memorisation be scored as generalisation.
"""

from __future__ import annotations

from kanal.data.dedup import (
    NUM_BANDS,
    NUM_PERM,
    ROWS_PER_BAND,
    Permutations,
    cluster,
    jaccard,
    normalise,
    shingles,
    signature,
)


class TestNormalise:
    def test_lowercases_and_strips_punctuation(self) -> None:
        assert normalise("Harga Emas Naik, Rupiah Menguat!") == "harga emas naik rupiah menguat"

    def test_collapses_whitespace(self) -> None:
        assert normalise("  a\n\n b\t c ") == "a b c"


class TestShingles:
    def test_produces_character_ngrams(self) -> None:
        assert shingles("abcdef", size=4) == {"abcd", "bcde", "cdef"}

    def test_a_short_title_yields_itself(self) -> None:
        # Returning an empty set would give it a degenerate signature that
        # matches everything.
        assert shingles("ab", size=4) == {"ab"}

    def test_empty_input_yields_nothing(self) -> None:
        assert shingles("") == set()

    def test_character_grams_survive_indonesian_affixes(self) -> None:
        # The reason for character rather than word n-grams: Indonesian is
        # agglutinative, and word tokens split a shared stem apart.
        a = shingles("pemerintah menaikkan cukai rokok")
        b = shingles("kenaikan cukai rokok oleh pemerintah")
        assert jaccard(a, b) > 0.35


class TestSignature:
    def test_is_deterministic_across_calls(self) -> None:
        perms = Permutations.seeded(42)
        title = "Harga emas menguat di tengah ketidakpastian"
        assert signature(title, perms) == signature(title, perms)

    def test_is_reproducible_from_the_seed(self) -> None:
        # A clustering that shifted between runs would make the frozen split
        # manifest worthless.
        title = "Timnas lolos ke babak berikutnya"
        assert signature(title, Permutations.seeded(7)) == signature(title, Permutations.seeded(7))

    def test_differs_with_a_different_seed(self) -> None:
        title = "Timnas lolos ke babak berikutnya"
        assert signature(title, Permutations.seeded(1)) != signature(title, Permutations.seeded(2))

    def test_has_one_entry_per_permutation(self) -> None:
        sig = signature("apa saja", Permutations.seeded())
        assert len(sig) == NUM_PERM
        assert NUM_BANDS * ROWS_PER_BAND == NUM_PERM

    def test_signature_agreement_approximates_jaccard(self) -> None:
        # The identity the whole method rests on: the share of positions where two
        # signatures agree estimates their Jaccard similarity.
        perms = Permutations.seeded(42)
        a = "Pemerintah umumkan kebijakan subsidi energi baru"
        b = "Pemerintah umumkan kebijakan subsidi energi terbaru"

        exact = jaccard(shingles(a), shingles(b))
        sig_a, sig_b = signature(a, perms), signature(b, perms)
        estimated = sum(x == y for x, y in zip(sig_a, sig_b, strict=True)) / NUM_PERM

        assert abs(estimated - exact) < 0.15


class TestClustering:
    def test_a_rewritten_headline_clusters_with_the_original(self) -> None:
        # The case the exact-hash layer in the warehouse cannot catch.
        rows = [
            ("a", "Pemerintah naikkan cukai rokok 10 persen tahun depan"),
            ("b", "Pemerintah naikkan cukai rokok 10 persen pada tahun depan"),
            ("c", "Timnas futsal menang atas Thailand di final"),
        ]
        report = cluster(rows)
        assert report.cluster_of["a"] == report.cluster_of["b"]
        assert report.cluster_of["c"] != report.cluster_of["a"]

    def test_identical_titles_cluster(self) -> None:
        rows = [("a", "Harga emas naik"), ("b", "Harga emas naik")]
        report = cluster(rows)
        assert report.cluster_of["a"] == report.cluster_of["b"]

    def test_unrelated_titles_stay_apart(self) -> None:
        rows = [
            ("a", "Bank sentral pertahankan suku bunga acuan"),
            ("b", "Timnas futsal menang atas Thailand di final"),
            ("c", "Peneliti temukan metode deteksi dini penyakit"),
        ]
        report = cluster(rows)
        assert len({report.cluster_of[k] for k in "abc"}) == 3

    def test_merging_is_transitive(self) -> None:
        # A~B and B~C must put all three together, or a split by cluster still
        # leaves related rows on both sides.
        rows = [
            ("a", "Presiden resmikan bendungan baru di Jawa Barat"),
            ("b", "Presiden resmikan bendungan baru di Jawa Timur"),
            ("c", "Presiden resmikan bendungan baru di Jawa Tengah"),
        ]
        report = cluster(rows)
        assert len({report.cluster_of[k] for k in "abc"}) == 1

    def test_cluster_ids_are_stable_under_reordering(self) -> None:
        rows = [
            ("a", "Harga emas menguat hari ini"),
            ("b", "Harga emas menguat pada hari ini"),
            ("c", "Kompetisi liga dimulai akhir pekan"),
        ]
        forward = cluster(rows).cluster_of
        backward = cluster(list(reversed(rows))).cluster_of
        assert forward == backward

    def test_every_row_gets_a_cluster(self) -> None:
        rows = [(str(i), f"Judul berita nomor {i}") for i in range(20)]
        report = cluster(rows)
        assert set(report.cluster_of) == {str(i) for i in range(20)}

    def test_a_singleton_is_its_own_cluster(self) -> None:
        report = cluster([("only", "Satu judul saja di sini")])
        assert report.cluster_of["only"] == "only"
        assert report.rows_in_a_cluster == 0

    def test_reports_what_it_found(self) -> None:
        rows = [
            ("a", "Pemerintah naikkan cukai rokok sepuluh persen"),
            ("b", "Pemerintah naikkan cukai rokok 10 persen"),
            ("c", "Sesuatu yang sama sekali berbeda tentang olahraga"),
        ]
        report = cluster(rows)
        assert report.candidate_pairs >= report.confirmed_pairs
        assert 0.0 <= report.false_candidate_rate <= 1.0
        assert "clusters over" in report.summary()

    def test_empty_input(self) -> None:
        report = cluster([])
        assert report.cluster_of == {}
        assert report.clusters == 0

    def test_lsh_is_a_filter_not_a_decision(self) -> None:
        # Banding produces false positives by design; exact Jaccard verification
        # is what keeps loosely-similar pairs from merging.
        rows = [
            ("a", "Harga minyak dunia turun tipis pada perdagangan Senin"),
            ("b", "Harga saham dunia naik tajam pada perdagangan Selasa"),
        ]
        report = cluster(rows)
        # Similar shape, different story — must not merge.
        assert report.cluster_of["a"] != report.cluster_of["b"]


class TestKnownLimitation:
    """Template collisions, which no threshold separates.

    Indonesian news headlines are heavily templated, and character n-grams over a
    shared template score as high as genuine duplicates. Measured on hand-labelled
    pairs: true duplicates 0.263-0.688, template collisions 0.379-0.575 —
    overlapping, so a threshold cannot tell them apart. Distinctive-token overlap
    and summary similarity were both measured and both failed too.

    These tests document the residual error rather than pretending it is absent,
    and pin the reasoning for tolerating it: the two error directions are not
    symmetric.
    """

    def test_a_shared_template_can_still_over_merge(self) -> None:
        # Two different fixtures, one headline form. This *does* merge, and that
        # is the accepted cost.
        rows = [
            ("a", "Link Live Streaming Port FC vs Persija di Piala Presiden 2026"),
            ("b", "Link Live Streaming Persebaya vs PSMS di Piala Presiden 2026"),
        ]
        report = cluster(rows)
        assert report.cluster_of["a"] == report.cluster_of["b"]

    def test_over_merging_is_the_safe_direction(self) -> None:
        # The asymmetry that justifies the threshold. Over-merging costs a little
        # test diversity; under-merging puts the same story in train and test and
        # invalidates the whole evaluation. So recall is favoured over precision.
        same_story = [
            ("a", "Serangan AS-Saudi ke Irak Tewaskan 20 Orang"),
            ("b", "Irak Kecam Serangan Gabungan AS-Saudi Tewaskan 20 Orang"),
        ]
        report = cluster(same_story)
        assert report.cluster_of["a"] == report.cluster_of["b"], (
            "a genuine duplicate must never be split across train and test"
        )

    def test_the_recall_ceiling_is_the_banding_not_the_threshold(self) -> None:
        """A heavily-rewritten headline is missed even below its own similarity.

        This pair has a Jaccard of 0.263, so lowering the threshold to 0.25 ought
        to merge it. It does not — because LSH never proposes it as a candidate in
        the first place.

        With 32 bands of 4 rows the S-curve knee sits near (1/32)^(1/4) ≈ 0.42, so
        a pair at 0.263 has only about a 14% chance of sharing any band. The
        threshold governs *verification*; the banding governs what is ever offered
        for verification, and it is the binding constraint here.

        Catching pairs this dissimilar would need re-banding (more bands, fewer
        rows per band), which floods the candidate set — already at a 98% false
        rate on the real corpus — for pairs that are as likely to be template
        collisions as duplicates. Recorded rather than tuned.
        """
        rewritten = [
            ("a", "Pemerintah naikkan cukai rokok 10 persen tahun depan"),
            ("b", "Cukai Rokok Naik 10 Persen pada 2027, Ini Alasannya"),
        ]
        report = cluster(rewritten, threshold=0.25)
        assert report.cluster_of["a"] != report.cluster_of["b"]
        # Nothing was even proposed, which is the point.
        assert report.candidate_pairs == 0

    def test_a_looser_threshold_merges_more_collisions(self) -> None:
        # Where LSH *does* surface a candidate, loosening the threshold merges it.
        # So the threshold trades precision away without buying back the recall
        # the banding lost — there is no setting that fixes both.
        collision = [
            ("c", "Daftar 2 Tim Lolos ke Semifinal Piala Presiden 2026"),
            ("d", "Syarat Persija Jakarta Lolos ke Semifinal Piala Presiden 2026"),
        ]
        report = cluster(collision, threshold=0.25)
        assert report.cluster_of["c"] == report.cluster_of["d"]

    def test_the_false_candidate_rate_is_reported(self) -> None:
        # LSH surfaces far more candidates than survive verification, and that
        # ratio is worth seeing: near zero means the bands are too tight to find
        # anything, near one means most of the work is wasted.
        rows = [(str(i), f"Judul berita yang berbeda nomor {i}") for i in range(30)]
        report = cluster(rows)
        assert 0.0 <= report.false_candidate_rate <= 1.0


class TestThreshold:
    def test_threshold_is_respected(self) -> None:
        rows = [
            ("a", "Pemerintah umumkan kebijakan baru soal subsidi energi"),
            ("b", "Pemerintah umumkan kebijakan baru soal subsidi listrik"),
        ]
        loose = cluster(rows, threshold=0.3)
        strict = cluster(rows, threshold=0.95)

        assert loose.cluster_of["a"] == loose.cluster_of["b"]
        assert strict.cluster_of["a"] != strict.cluster_of["b"]
