"""_resolve_footnotes: chunk markers -> markdown footnotes.

The prompt hands the model short handles ([C1], [C2], ...) rather than raw
chunk ids, because models abbreviate a 40-char hash (`chunk-b080ee8...-000003`)
and the truncated marker then resolves to nothing and leaks to the user.
The chunk-id path is kept as a fallback and still tested here,
including the full-width-bracket drift (【…】) that weaker models produce
instead of the ASCII [ ] the prompt asks for.
"""
import asyncio
import importlib.util
import pathlib
import re

from src_rag_web import mcp_route

# Loaded by path, not as src_rag_web.utils.helpers: the utils package __init__
# pulls in the heavy import chain (bs4, torch, ...) that the slim test image
# deliberately omits. helpers.py itself is stdlib-only.
_spec = importlib.util.spec_from_file_location(
    "_helpers_under_test",
    pathlib.Path(__file__).resolve().parent.parent / "src_rag_web/utils/helpers.py",
)
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
convert_to_user_format = _helpers.convert_to_user_format
generate_reference_list_from_chunks = _helpers.generate_reference_list_from_chunks

PAPER_ID = "6e12b7c9a1d3f5074b2e8c6a9f0d1b3e5c7a9d20"
CHUNK_ID = f"chunk-{PAPER_ID}-000002"

PAPER_ID_2 = "1a2b3c4d5e6f70819a2b3c4d5e6f708192a3b4c5"
CHUNK_ID_2 = f"chunk-{PAPER_ID_2}-000004"

META = {
    PAPER_ID: {
        "title": "BRCA1 and homologous recombination repair",
        "bibtex_json": {"authors": ["Doe J", "Roe A"], "year": "2021"},
        "journal": {"name": "Nature"},
        "doi": "10.1000/example",
    },
    PAPER_ID_2: {
        "title": "PTEN antagonizes PI3K signaling",
        "bibtex_json": {"authors": ["Poe E"], "year": "2019"},
        "journal": {"name": "Cell"},
    },
}


class _FakeMongo:
    async def get_papers_metadata(self, paper_ids):
        return {pid: META[pid] for pid in paper_ids if pid in META}


class _FakeRagApp:
    mongo_storage = _FakeMongo()


def _resolve(response_text, raw_refs=None):
    if raw_refs is None:
        raw_refs = [
            {"id": CHUNK_ID, "handle": "C1", "content": "BRCA1 promotes HR repair."}
        ]
    return asyncio.run(
        mcp_route._resolve_footnotes(response_text, raw_refs, _FakeRagApp())
    )


TWO_REFS = [
    {"id": CHUNK_ID, "handle": "C1", "content": "BRCA1 promotes HR repair."},
    {"id": CHUNK_ID_2, "handle": "C2", "content": "PTEN converts PIP3 to PIP2."},
]


def test_fullwidth_cjk_marker_becomes_footnote():
    """The regression: 【…】 used to pass through as literal text."""
    result = _resolve(f"BRCA1 promotes HR repair 【{CHUNK_ID}】.")

    assert "BRCA1 promotes HR repair [^1]." in result
    assert "【" not in result and "】" not in result
    assert CHUNK_ID not in result.split("[^1]:")[0]  # no raw marker in the body
    assert "[^1]: Doe J, Roe A (2021)." in result


def test_ascii_marker_still_becomes_footnote():
    result = _resolve(f"BRCA1 promotes HR repair [{CHUNK_ID}].")

    assert "BRCA1 promotes HR repair [^1]." in result
    assert "[^1]: Doe J, Roe A (2021)." in result


def test_fullwidth_ascii_brackets_become_footnote():
    """U+FF3B/U+FF3D — the other half of the same drift class."""
    result = _resolve(f"BRCA1 promotes HR repair ［{CHUNK_ID}］.")

    assert "BRCA1 promotes HR repair [^1]." in result
    assert "［" not in result and "］" not in result


def test_uppercase_hash_resolves_rather_than_being_dropped():
    upper = f"chunk-{PAPER_ID.upper()}-000002"
    result = _resolve(f"BRCA1 promotes HR repair 【{upper}】.")

    assert "BRCA1 promotes HR repair [^1]." in result
    assert upper not in result


def test_unrelated_fullwidth_brackets_are_left_alone():
    """Normalization is scoped to chunk markers, not every 【…】 in the answer."""
    result = _resolve(f"As shown in 【Table 1】, BRCA1 promotes HR [{CHUNK_ID}].")

    assert "As shown in 【Table 1】, BRCA1 promotes HR [^1]." in result


def test_fullwidth_marker_is_stripped_when_there_are_no_refs():
    """The no-refs fallback path must strip full-width markers too."""
    result = _resolve(f"BRCA1 promotes HR repair 【{CHUNK_ID}】.", raw_refs=[])

    assert result == "BRCA1 promotes HR repair ."
    assert "chunk-" not in result


# --- short handles -----------------------------------------------------------


def test_short_handle_becomes_footnote():
    result = _resolve("BRCA1 promotes HR repair [C1].")

    assert "BRCA1 promotes HR repair [^1]." in result
    assert "[^1]: Doe J, Roe A (2021)." in result
    assert "[C1]" not in result


def test_fullwidth_handle_becomes_footnote():
    """Full-width drift applies to handles too, not just chunk ids."""
    for text in (f"BRCA1 promotes HR repair 【C1】.", f"BRCA1 promotes HR repair ［C1］."):
        result = _resolve(text)

        assert "BRCA1 promotes HR repair [^1]." in result
        assert "【" not in result and "［" not in result


def test_lowercase_handle_resolves():
    result = _resolve("BRCA1 promotes HR repair [c1].")

    assert "BRCA1 promotes HR repair [^1]." in result


def test_grouped_handles_are_split_into_separate_footnotes():
    """Models sometimes write [C1, C2] instead of [C1][C2]."""
    result = _resolve("Both hold [C1, C2].", raw_refs=TWO_REFS)

    assert "Both hold [^1][^2]." in result
    assert "[^1]: Doe J, Roe A (2021)." in result
    assert "[^2]: Poe E (2019)." in result


def test_separate_handles_resolve_and_sort():
    result = _resolve("Both hold [C2][C1].", raw_refs=TWO_REFS)

    assert "Both hold [^1][^2]." in result


def test_unknown_handle_is_stripped_not_leaked():
    """An invented handle must not reach the user as literal text."""
    result = _resolve("BRCA1 promotes HR repair [C9]. PTEN too [C1].")

    assert "[C9]" not in result
    assert "BRCA1 promotes HR repair . PTEN too [^1]." in result


def test_handle_is_stripped_when_there_are_no_refs():
    result = _resolve("BRCA1 promotes HR repair [C1].", raw_refs=[])

    assert result == "BRCA1 promotes HR repair ."
    assert "C1" not in result


def _numbered_corpus(n_good=6, n_maybe=14, cap=10):
    """Chunks whose text and paper title both encode their identity.

    Mirrors build_llm_context's mix-mode split so the good/maybe section
    boundary and the top-10 maybe cap are exercised for real.
    """
    def make(i, score):
        pid = f"{i:040x}"  # injective: one distinct paper per chunk
        return {
            "id": f"chunk-{pid}-{i:06d}",
            "rerank_score": score,
            "content": f"SENTINEL-{i:03d} is the evidence for chunk {i}.",
            "_paper": pid,
        }

    chunks = [make(i, 0.9 - i * 0.01) for i in range(n_good)]
    chunks += [make(100 + i, 0.4 - i * 0.01) for i in range(n_maybe)]
    meta = {
        c["_paper"]: {
            "title": f"TITLE-{c['id'][-6:]}",
            "bibtex_json": {"authors": [f"Author{c['id'][-3:]}"], "year": "2024"},
            "journal": {"name": "Journal"},
        }
        for c in chunks
    }

    good = [c for c in chunks if c["rerank_score"] >= 0.5]
    maybe = [c for c in chunks if c["rerank_score"] < 0.5]
    maybe.sort(key=lambda x: x["rerank_score"], reverse=True)
    maybe = maybe[:cap]
    good.sort(key=lambda x: x["id"])
    maybe.sort(key=lambda x: x["id"])

    prompt = (
        convert_to_user_format(good)
        + "\n\n"
        + convert_to_user_format(maybe, start_index=len(good) + 1)
    )
    return chunks, generate_reference_list_from_chunks(chunks), meta, prompt


def test_citations_keep_their_position_and_identity():
    """Each [Cn] must become a [^m] in the same slot, pointing at its own chunk.

    Citations are laid out awkwardly on purpose — descending, repeated,
    mid-sentence, adjacent — so an off-by-one or an ordering bug cannot hide.
    """
    chunks, refs, meta, prompt = _numbered_corpus()

    # What the model actually sees: handle -> the sentinel under that header.
    shown = dict(
        block.split("\n", 1) for block in prompt.split("\n\n") if block.startswith("[C")
    )
    sentinel_of = {
        h.strip("[]"): re.search(r"SENTINEL-\d+", body).group(0)
        for h, body in shown.items()
    }
    assert sorted(sentinel_of, key=lambda h: int(h[1:])) == [
        f"C{i}" for i in range(1, 17)
    ]  # 6 good + 10 surviving the cap

    class _CorpusMongo:
        async def get_papers_metadata(self, paper_ids):
            return {p: meta[p] for p in paper_ids if p in meta}

    class _CorpusRagApp:
        mongo_storage = _CorpusMongo()

    answer = (
        "Opening claim with a late source [C12]. "
        "Second claim cites two, reversed [C7][C3]. "
        "Third repeats an earlier one [C12] mid-sentence, then ends [C1]. "
        "Fourth cites a maybe-related chunk [C9] and a good one [C2]."
    )
    out = asyncio.run(mcp_route._resolve_footnotes(answer, refs, _CorpusRagApp()))
    body, _, defs = out.partition("\n\n[^1]:")
    defs = "[^1]:" + defs

    # Position: swap every marker for a placeholder — the sentences must match.
    assert re.sub(r"\[\^\d+\]", "<REF>", body) == re.sub(r"\[C\d+\]", "<REF>", answer)

    # Identity: pair markers in text order, then check each definition quotes
    # the chunk its handle named and cites that chunk's paper.
    id_of = {r["handle"]: r["id"] for r in refs if r["handle"]}
    num_to_handle = {}
    for handle, num in zip(
        re.findall(r"\[C(\d+)\]", answer), re.findall(r"\[\^(\d+)\]", body)
    ):
        assert num_to_handle.setdefault(num, handle) == handle
    for block in re.split(r"\n\n(?=\[\^\d+\]:)", defs):
        num = re.match(r"\[\^(\d+)\]:", block).group(1)
        handle = "C" + num_to_handle[num]
        assert sentinel_of[handle] in block
        assert f"TITLE-{id_of[handle][-6:]}" in block

    # Numbering: dense, ascending by first appearance, one definition each.
    seq = list(dict.fromkeys(re.findall(r"\[\^(\d+)\]", body)))
    assert [int(n) for n in seq] == list(range(1, len(seq) + 1))
    assert len(re.findall(r"^\[\^\d+\]:", defs, re.M)) == len(seq)
    assert "[C" not in out and "chunk-" not in body


def test_chunks_beyond_the_cap_get_no_handle():
    """Chunks the model never saw must not be citable, and must not shift others."""
    _chunks, refs, _meta, _prompt = _numbered_corpus()

    assert sum(1 for r in refs if r["handle"]) == 16
    assert sum(1 for r in refs if not r["handle"]) == 4  # 14 maybe - 10 cap


def test_colocated_refs_are_normalized_ascending():
    """Adjacent refs are sorted, so within-group order is not the model's.

    Pre-existing behaviour (_sort_ref_group): the set at each position is
    preserved, only the order inside the bracket run is normalized.
    """
    result = _resolve("First [C2]. Later [C1][C2] together.", raw_refs=TWO_REFS)

    # C2 cited first -> [^1]; C1 -> [^2]; the [C1][C2] run renders ascending.
    assert "First [^1]. Later [^1][^2] together." in result


def test_prompt_handles_match_reference_handles():
    """The handle the model sees must be the handle the resolver maps back."""
    good = [{"id": CHUNK_ID, "content": "BRCA1 promotes HR repair."}]
    maybe = [{"id": CHUNK_ID_2, "content": "PTEN converts PIP3 to PIP2."}]

    good_text = convert_to_user_format(good)
    maybe_text = convert_to_user_format(maybe, start_index=len(good) + 1)

    assert good_text.startswith("[C1]\nBRCA1")
    assert maybe_text.startswith("[C2]\nPTEN")
    assert CHUNK_ID not in good_text  # no raw hash for the model to truncate

    refs = generate_reference_list_from_chunks(good + maybe)
    assert [(r["handle"], r["id"]) for r in refs] == [
        ("C1", CHUNK_ID),
        ("C2", CHUNK_ID_2),
    ]

    result = _resolve("BRCA1 [C1]. PTEN [C2].", raw_refs=refs)
    assert "BRCA1 [^1]. PTEN [^2]." in result


def test_chunk_without_handle_is_ignored():
    """A chunk the model never saw has handle=None and must not break the map."""
    refs = [
        {"id": CHUNK_ID, "handle": None, "content": "unseen"},
        {"id": CHUNK_ID_2, "handle": "C1", "content": "PTEN converts PIP3 to PIP2."},
    ]
    result = _resolve("PTEN converts PIP3 [C1].", raw_refs=refs)

    assert "PTEN converts PIP3 [^1]." in result
    assert "[^1]: Poe E (2019)." in result


# --- malformed inputs must degrade, not raise --------------------------------


def test_malformed_metadata_still_produces_a_footnote():
    """journal as a bare string + authors as dicts must not raise a 500."""
    paper_id = "0f1e2d3c4b5a69788796a5b4c3d2e1f009182736"
    chunk_id = f"chunk-{paper_id}-000001"

    class _MalformedMongo:
        async def get_papers_metadata(self, paper_ids):
            return {
                paper_id: {
                    "title": "A paper with an awkward record",
                    "bibtex_json": {
                        "authors": [{"family": "Doe", "given": "J"}, {"name": "Roe A"}],
                        "year": 2020,
                    },
                    "journal": "Science",
                }
            }

    class _MalformedRagApp:
        mongo_storage = _MalformedMongo()

    result = asyncio.run(
        mcp_route._resolve_footnotes(
            "A claim [C1].",
            [{"id": chunk_id, "handle": "C1", "content": "Evidence."}],
            _MalformedRagApp(),
        )
    )

    assert "A claim [^1]." in result
    assert "A paper with an awkward record" in result
    assert "Doe J, Roe A (2020)" in result
    assert "*Science*" in result


def test_ref_with_none_id_does_not_crash():
    """A chunk stored with id=None used to raise TypeError in re.match."""
    refs = [
        {"id": None, "handle": "C1", "content": "no id"},
        {"id": CHUNK_ID, "handle": "C2", "content": "BRCA1 promotes HR repair."},
    ]
    result = _resolve("Unknown [C1]. BRCA1 promotes HR repair [C2].", raw_refs=refs)

    assert "BRCA1 promotes HR repair [^1]." in result
    assert "[C1]" not in result


# --- same-paper chunk grouping -----------------------------------------------
#
# Several chunks of ONE paper used to produce several full bibliographic entries
# that differed only in their excerpt. They now share a paper index and are
# labelled P.C, which is what lets the renderer merge them into one entry with
# nested excerpts. The label is only an identifier — GFM still DISPLAYS 1, 2,
# 3… — so these tests assert the markdown, not what a given renderer displays.

# Three chunks of PAPER_ID, plus one of PAPER_ID_2.
CHUNK_ID_B = f"chunk-{PAPER_ID}-000007"
CHUNK_ID_C = f"chunk-{PAPER_ID}-000009"

GROUPED_REFS = [
    {"id": CHUNK_ID, "handle": "C1", "content": "BRCA1 promotes HR repair."},
    {"id": CHUNK_ID_B, "handle": "C2", "content": "RAD51 loading needs BRCA2."},
    {"id": CHUNK_ID_C, "handle": "C3", "content": "53BP1 antagonizes resection."},
    {"id": CHUNK_ID_2, "handle": "C4", "content": "PTEN converts PIP3 to PIP2."},
]


def test_same_paper_chunks_share_a_paper_index():
    """TEST-1: 3 chunks of one paper -> 1-1/1-2/1-3; a 1-chunk paper stays FLAT.

    The hierarchy appears only where it carries information, so the paper cited
    through a single chunk is "2", not "2-1". The renderer is what keeps that
    flat label from displaying as its ordinal position (3) — see
    stampFootnoteRefLabels.
    """
    result = _resolve("HR repair [C1][C2][C3]. PTEN [C4].", raw_refs=GROUPED_REFS)

    body, _, defs = result.partition("\n\n[^")
    assert "HR repair [^1-1][^1-2][^1-3]. PTEN [^2]." in body

    # One definition per chunk, each carrying the full bib header — a renderer
    # that does NOT group still shows a complete reference (graceful degradation).
    defs = "[^" + defs
    assert defs.count("Doe J, Roe A (2021)") == 3
    for label, excerpt in [
        ("1-1", "BRCA1 promotes HR repair."),
        ("1-2", "RAD51 loading needs BRCA2."),
        ("1-3", "53BP1 antagonizes resection."),
    ]:
        assert f"[^{label}]: Doe J, Roe A (2021)." in defs
        assert f"> {excerpt}" in defs

    assert "[^2]: Poe E (2019)." in defs


def test_all_single_chunk_papers_stay_flat():
    """TEST-1c: with no paper grouped, labels stay flat — today's exact output."""
    result = _resolve("A [C1]. B [C4].", raw_refs=GROUPED_REFS)

    assert "A [^1]. B [^2]." in result
    assert "." not in "".join(re.findall(r"\[\^([\d.]+)\]", result))


def test_papers_are_numbered_by_first_appearance_even_when_interleaved():
    """TEST-1b: citation order 1.1, 2, 1.2 still yields two dense paper indices."""
    result = _resolve("A [C1]. B [C4]. C [C2].", raw_refs=GROUPED_REFS)

    assert "A [^1-1]. B [^2]. C [^1-2]." in result
    # Definitions are emitted grouped by paper, not in citation order.
    assert result.index("[^1-1]: ") < result.index("[^1-2]: ") < result.index("[^2]: ")


def test_chunk_without_metadata_does_not_corrupt_paper_numbering():
    """TEST-2: a dropped chunk consumes no paper index and no chunk index."""
    unknown = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    refs = [
        # A distinct handle: GROUPED_REFS already owns C1-C4.
        {"id": f"chunk-{unknown}-000001", "handle": "C9", "content": "orphan"},
        *GROUPED_REFS,
    ]
    result = _resolve("Orphan [C9]. HR [C2][C3]. PTEN [C4].", raw_refs=refs)

    # C9 is dropped entirely; the surviving papers are still 1 and 2, and the
    # two chunks of paper 1 are still .1 and .2 — no gap, no shift.
    assert "Orphan . HR [^1-1][^1-2]. PTEN [^2]." in result
    assert unknown not in result


def test_paper_without_doi_still_groups_by_hash():
    """TEST-3: the grouping key is the paper hash, never the DOI."""
    # PAPER_ID_2 has no "doi" key in META.
    refs = [
        {"id": CHUNK_ID_2, "handle": "C1", "content": "PTEN converts PIP3 to PIP2."},
        {
            "id": f"chunk-{PAPER_ID_2}-000011",
            "handle": "C2",
            "content": "PTEN loss activates AKT.",
        },
    ]
    result = _resolve("A [C1]. B [C2].", raw_refs=refs)

    assert "A [^1-1]. B [^1-2]." in result
    assert result.count("Poe E (2019)") == 2
    assert "https://doi.org/" not in result


def test_colocated_hierarchical_refs_sort_numerically_and_close_up():
    """TEST-4: 1.10 sorts after 1.2, and whitespace inside a run is closed up.

    The renderer separates adjacent citation superscripts with an adjacent-
    sibling CSS rule, so a run must end up with no gap between the markers.
    """
    assert mcp_route._label_sort_key("1-10") > mcp_route._label_sort_key("1-2")

    # Ten chunks of one paper so labels reach 1.10, cited out of order and spaced.
    refs = [
        {
            "id": f"chunk-{PAPER_ID}-{i:06d}",
            "handle": f"C{i}",
            "content": f"Chunk {i}.",
        }
        for i in range(1, 11)
    ]
    cite_in_order = "".join(f"[C{i}]" for i in range(1, 11))
    result = _resolve(f"All {cite_in_order}. Again [C10] [C2].", raw_refs=refs)

    assert "Again [^1-2][^1-10]." in result

    # A run spanning a newline is NOT merged — only spaces/tabs are closed up.
    # C2 is cited first so it takes .1 and C1 takes .2; the pair across the
    # newline is therefore DESCENDING, and stays descending (unsorted, unmerged).
    spaced = _resolve("A [C2]. X [C1]\n[C2] B.", raw_refs=refs)
    assert "X [^1-2]\n[^1-1] B." in spaced
