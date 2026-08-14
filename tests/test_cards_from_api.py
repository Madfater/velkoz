import json
from types import SimpleNamespace

import httpx
import pytest

from riftbound_bot.ingest.cards_from_api import (
    API_CARDS_URL,
    TRANSLATION_ATTEMPTS,
    ApiCard,
    TranslationBatchError,
    _clean_name,
    _clean_text,
    _dedupe_printings,
    _normalize_card,
    _parse_batch_response,
    _translate_batch,
    fetch_api_cards,
)
from riftbound_bot.ingest.http import is_retryable


def test_rules_lines_are_split_on_rich_text_breaks():
    # `text.plain` drops these breaks without substituting anything, running
    # one ability into the next — which is why the rich field is the source.
    raw = (
        "<p>[Deflect] (Opponents must pay :rb_rune_rainbow: to choose me.)<br />"
        "When you play me, gain 3 XP.</p>"
    )
    assert _clean_text(raw) == (
        "[Deflect] (Opponents must pay any rune to choose me.)\n"
        "When you play me, gain 3 XP."
    )


def test_icon_tokens_expand_to_words_without_stray_spacing():
    # The expansion pads its replacements, so it has to tidy up the space it
    # would otherwise leave sitting in front of punctuation.
    # Verbatim cost line from UNL-026 Xerath - Freed.
    raw = "<p>:rb_rune_fury:, :rb_exhaust:: Deal 3 to a unit with 2 :rb_might:.</p>"
    assert _clean_text(raw) == "Fury rune, Exhaust: Deal 3 to a unit with 2 Might."


def test_energy_tokens_and_html_entities_are_resolved():
    raw = "<p>[Deathknell][&gt;] Pay :rb_energy_2: to draw 1.</p>"
    assert _clean_text(raw) == "[Deathknell][>] Pay 2 Energy to draw 1."


def test_list_markup_becomes_separate_lines():
    raw = "<p>Choose one:</p><ul><li>Draw 1.</li><li>Deal 1.</li></ul>"
    assert _clean_text(raw) == "Choose one:\nDraw 1.\nDeal 1."


def test_printing_qualifiers_are_stripped_from_names():
    assert _clean_name("Vi - Piltover Enforcer (Signature)") == "Vi - Piltover Enforcer"
    assert _clean_name("Poppy - Paragon (Alternate Art)") == "Poppy - Paragon"


def test_parentheses_that_are_part_of_the_name_are_kept():
    # Token cards are numbered in their own name; only the known printing
    # qualifiers get stripped, so these must survive intact.
    assert _clean_name("Sprite (274) // Buff") == "Sprite (274) // Buff"
    assert _clean_name("Recruit (273) // Buff") == "Recruit (273) // Buff"


def _raw_card(**overrides):
    raw = {
        "name": "Bewitching Spirit",
        "collector_number": 121,
        "set": {"set_id": "UNL", "label": "Unleashed"},
        "classification": {"type": "Unit", "rarity": "Common", "domain": ["Chaos"]},
        "attributes": {"energy": 3, "might": 2, "power": None},
        "text": {"rich": "<p>When you play me, choose a player. They discard 1.</p>"},
        "tags": ["Spirit", "Shadow Isles"],
        "metadata": {},
    }
    raw.update(overrides)
    return raw


def test_normalize_maps_onto_the_primary_sources_vocabulary():
    # Both sources write the same columns, so the fallback has to emit the
    # same values chroniclecore does: an English colour word, Chinese
    # category/rarity/trait tags, and a `SET-NNN` id.
    card = _normalize_card(_raw_card())
    assert card.id == "UNL-121"
    assert card.set == "UNL"
    assert card.collector_number == "121"
    assert card.color == "purple"
    assert card.category == "單位"
    assert card.rarity == "普通"
    assert card.tags == ["靈體", "暗影島"]
    assert (card.energy, card.might, card.power) == (3, 2, None)


def test_dual_domain_cards_keep_both_colours():
    card = _normalize_card(
        _raw_card(classification={"type": "Legend", "rarity": "Rare", "domain": ["Fury", "Chaos"]})
    )
    assert card.color == "red/purple"


def test_champion_name_tags_pass_through_untranslated():
    # Only the trait vocabulary has a verified mapping; champion names are
    # left alone rather than guessed at.
    card = _normalize_card(_raw_card(tags=["Vi", "Piltover"]))
    assert card.tags == ["Vi", "皮爾特沃夫"]


def test_dedupe_keeps_the_base_printing():
    # The base printing carries the parenthetical reminder text that the
    # alternate-art printing omits, so it is the one worth indexing.
    base = _raw_card(metadata={"alternate_art": False})
    alt = _raw_card(name="Bewitching Spirit (Alternate Art)", metadata={"alternate_art": True})
    assert _dedupe_printings([alt, base]) == [base]
    assert _dedupe_printings([base, alt]) == [base]


def test_dedupe_collapses_rows_sharing_a_collector_number():
    metal = _raw_card(name="Bewitching Spirit (Metal)", tcgplayer_id="2")
    plain = _raw_card(tcgplayer_id="1")
    other = _raw_card(collector_number=122, name="Something Else")
    deduped = _dedupe_printings([metal, plain, other])
    assert len(deduped) == 2
    assert {_normalize_card(c).id for c in deduped} == {"UNL-121", "UNL-122"}
    # Which row survives is the point of the collapse, not just how many.
    assert plain in deduped and metal not in deduped


def test_dedupe_prefers_the_base_printing_over_a_name_only_qualifier():
    # Metal/Starter/Ultimate carry no metadata flag at all — the qualifier is
    # only visible in the trailing parenthetical of the name.
    metal = _raw_card(name="Bewitching Spirit (Metal)")
    plain = _raw_card()
    assert _dedupe_printings([metal, plain]) == [plain]
    assert _dedupe_printings([plain, metal]) == [plain]


def test_dedupe_tiebreak_compares_tcgplayer_ids_numerically():
    # As strings "10" sorts before "9", picking the wrong printing.
    lower = _raw_card(tcgplayer_id="9")
    higher = _raw_card(tcgplayer_id="10")
    assert _dedupe_printings([higher, lower]) == [lower]


def test_dedupe_prefers_a_printing_that_has_a_tcgplayer_id():
    identified = _raw_card(tcgplayer_id="7")
    unidentified = _raw_card(tcgplayer_id=None)
    assert _dedupe_printings([unidentified, identified]) == [identified]


def test_collector_number_zero_is_padded_not_blanked():
    # `or ""` treated 0 as absent, so the id became UNL-000 by accident
    # rather than by padding.
    assert _normalize_card(_raw_card(collector_number=0)).collector_number == "000"


def test_cards_without_rules_text_normalize_to_empty_string():
    # Vanilla units have no rules text at all; that is not an error.
    assert _normalize_card(_raw_card(text={})).text_en == ""


def _api_card(card_id: str, name_en: str = "Bewitching Spirit") -> ApiCard:
    return ApiCard(
        id=card_id,
        name_en=name_en,
        text_en="When you play me, choose a player. They discard 1.",
        set=card_id.split("-")[0],
        collector_number=card_id.split("-")[1],
        category="單位",
        color="purple",
        rarity="普通",
        energy=3,
        power=None,
        might=2,
    )


def _translated(card_id: str, name_zh: str = "魅惑之靈", text_zh: str = "效果") -> dict:
    return {"id": card_id, "name_zh": name_zh, "text_zh": text_zh}


class _ScriptedLLM:
    """Returns each queued response in turn, recording how often it was asked."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return SimpleNamespace(content=self.responses.pop(0))


def test_batch_response_parses_ids_to_translations():
    batch = [_api_card("UNL-121"), _api_card("UNL-122")]
    content = json.dumps([_translated("UNL-121"), _translated("UNL-122", "另一張")])

    assert _parse_batch_response(content, batch) == {
        "UNL-121": ("魅惑之靈", "效果"),
        "UNL-122": ("另一張", "效果"),
    }


def test_batch_response_survives_markdown_code_fences():
    # The prompt forbids them, but models add them anyway.
    batch = [_api_card("UNL-121")]
    content = f"```json\n{json.dumps([_translated('UNL-121')])}\n```"

    assert _parse_batch_response(content, batch)["UNL-121"] == ("魅惑之靈", "效果")


def test_batch_response_with_wrong_ids_is_rejected():
    # The damaging case: a response that parses fine but describes other
    # cards used to leave every card in the batch holding English text.
    batch = [_api_card("UNL-121")]
    content = json.dumps([_translated("OGN-001")])

    with pytest.raises(TranslationBatchError, match="ids don't match"):
        _parse_batch_response(content, batch)


def test_batch_response_missing_a_card_is_rejected():
    batch = [_api_card("UNL-121"), _api_card("UNL-122")]
    content = json.dumps([_translated("UNL-121")])

    with pytest.raises(TranslationBatchError, match="missing \\['UNL-122'\\]"):
        _parse_batch_response(content, batch)


def test_batch_response_missing_a_key_is_rejected():
    batch = [_api_card("UNL-121")]
    content = json.dumps([{"id": "UNL-121", "name_zh": "魅惑之靈"}])

    with pytest.raises(TranslationBatchError, match="text_zh"):
        _parse_batch_response(content, batch)


def test_non_json_batch_response_is_rejected():
    with pytest.raises(TranslationBatchError, match="not valid JSON"):
        _parse_batch_response("I'm afraid I can't do that", [_api_card("UNL-121")])


def test_translate_batch_retries_an_invalid_response_then_succeeds():
    batch = [_api_card("UNL-121")]
    llm = _ScriptedLLM(["not json", json.dumps([_translated("UNL-121")])])

    assert _translate_batch(llm, batch) == {"UNL-121": ("魅惑之靈", "效果")}
    assert llm.calls == 2


def test_translate_batch_raises_once_attempts_are_exhausted():
    # Failing loudly beats writing English into name_zh and moving on.
    batch = [_api_card("UNL-121")]
    llm = _ScriptedLLM(["not json"] * TRANSLATION_ATTEMPTS)

    with pytest.raises(TranslationBatchError):
        _translate_batch(llm, batch)
    assert llm.calls == TRANSLATION_ATTEMPTS


class _FakePager:
    """Stands in for the paged /cards endpoint."""

    def __init__(self, pages: list[dict]):
        self.pages = pages
        self.requested: list[int] = []

    def __call__(self, client, page):
        self.requested.append(page)
        return self.pages[page - 1]


def _page(items: list[dict], pages: int | None) -> dict:
    payload: dict = {"items": items}
    if pages is not None:
        payload["pages"] = pages
    return payload


def test_fetch_pages_through_the_whole_list(monkeypatch):
    pager = _FakePager([
        _page([_raw_card(collector_number=1)], pages=2),
        _page([_raw_card(collector_number=2)], pages=2),
    ])
    monkeypatch.setattr("riftbound_bot.ingest.cards_from_api._fetch_page", pager)

    cards = fetch_api_cards()

    assert pager.requested == [1, 2]
    assert {c.id for c in cards} == {"UNL-001", "UNL-002"}


def test_fetch_stops_on_an_empty_page_rather_than_looping(monkeypatch):
    # A `pages` count larger than the data would otherwise keep requesting.
    pager = _FakePager([
        _page([_raw_card(collector_number=1)], pages=99),
        _page([], pages=99),
    ])
    monkeypatch.setattr("riftbound_bot.ingest.cards_from_api._fetch_page", pager)

    assert {c.id for c in fetch_api_cards()} == {"UNL-001"}
    assert pager.requested == [1, 2]


def test_fetch_without_a_page_count_does_not_silently_truncate(monkeypatch):
    # `pages` missing used to mean "one page", quietly returning the first
    # 100 of ~1,131 cards as though that were the whole corpus.
    pager = _FakePager([
        _page([_raw_card(collector_number=1)], pages=None),
        _page([_raw_card(collector_number=2)], pages=None),
        _page([], pages=None),
    ])
    monkeypatch.setattr("riftbound_bot.ingest.cards_from_api._fetch_page", pager)

    assert {c.id for c in fetch_api_cards()} == {"UNL-001", "UNL-002"}


def test_retry_policy_covers_transport_and_server_errors_only():
    request = httpx.Request("GET", API_CARDS_URL)

    def status_error(code: int) -> httpx.HTTPStatusError:
        response = httpx.Response(status_code=code, request=request)
        return httpx.HTTPStatusError("boom", request=request, response=response)

    assert is_retryable(httpx.ConnectError("down", request=request))
    assert is_retryable(status_error(503))
    assert is_retryable(status_error(429))
    # The two documented failure modes of this API: retrying them just repeats
    # the same deterministic rejection four times.
    assert not is_retryable(status_error(403))
    assert not is_retryable(status_error(422))
