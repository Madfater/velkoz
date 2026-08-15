from riftbound_bot.ingest.cards_from_api import (
    _clean_name,
    _clean_text,
    _dedupe_printings,
    _normalize_card,
)


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


def test_normalize_carries_the_media_image_url_through():
    # Already absolute here (Riot's CDN), unlike the primary scraper's
    # site-relative asset paths — both sources have to land an image in the
    # same column, because /card refuses to render a card without one.
    card = _normalize_card(
        _raw_card(media={"image_url": "https://cmsassets.rgpub.io/x-744x1039.png"})
    )
    assert card.image_url == "https://cmsassets.rgpub.io/x-744x1039.png"


def test_normalize_leaves_the_image_empty_when_the_api_omits_media():
    assert _normalize_card(_raw_card()).image_url == ""


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


def test_cards_without_rules_text_normalize_to_empty_string():
    # Vanilla units have no rules text at all; that is not an error.
    assert _normalize_card(_raw_card(text={})).text_en == ""
