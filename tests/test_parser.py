"""Tests for the rule-based request parser and the SearchQuery/Range types."""

import pytest

from src.parser import HIGH_ENERGY, LOW_ENERGY, LOW_VALENCE, parse_request
from src.query import Range, SearchQuery


def test_golden_path_request():
    q = parse_request("recommend something chill, with jazz and indie music")
    assert q.genres == ["jazz", "indie"]
    assert q.energy == LOW_ENERGY
    assert "chill" in q.moods
    assert q.unknown_genres == [] and q.exclude_genres == []


def test_longest_alias_wins():
    q = parse_request("indie pop and hip hop")
    assert q.genres == ["indie-pop", "hip-hop"]   # not indie, pop or "hop"


def test_aliases_map_to_catalog_genres():
    assert parse_request("some R&B and alternative rock").genres == ["r-n-b", "alt-rock"]


def test_negated_genre_is_excluded():
    q = parse_request("high energy workout metal, no rap")
    assert q.genres == ["metal"]
    assert q.exclude_genres == ["hip-hop"]
    assert q.energy == HIGH_ENERGY


def test_high_energy_phrase_not_double_counted():
    assert parse_request("high energy rock").moods == ["high energy"]


def test_low_energy_phrase_is_not_high_energy():
    assert parse_request("low energy jazz").energy == LOW_ENERGY


def test_unknown_genre_gets_note_with_closest_match():
    q = parse_request("k-pop dance hits")
    assert q.unknown_genres == ["k-pop"]
    assert q.genres == ["dance"]                  # "pop" inside "k-pop" isn't read as pop
    assert "closest genre is 'pop'" in q.notes[0]


def test_conflicting_moods_leave_energy_unfiltered():
    q = parse_request("calm but energetic music")
    assert q.energy is None
    assert any("both calm and energetic" in note for note in q.notes)


def test_sad_acoustic_sets_valence_and_acoustic():
    q = parse_request("sad acoustic songs for a rainy day")
    assert q.valence == LOW_VALENCE
    assert q.acoustic is True


def test_no_vocals_means_instrumental_not_excluded_genre():
    q = parse_request("piano with no vocals")
    assert q.instrumental is True
    assert q.genres == ["piano"] and q.exclude_genres == []


def test_song_count_in_request_sets_k():
    assert parse_request("give me 8 songs of jazz").k == 8


def test_k_is_clamped():
    assert parse_request("give me 99 songs of jazz").k == 20


def test_empty_request_rejected():
    with pytest.raises(ValueError):
        parse_request("   ")


def test_word_boundaries_avoid_false_matches():
    q = parse_request("an emotional trap anthem")   # "emo" in emotional, "rap" in trap
    assert q.unknown_genres == [] and q.genres == []


def test_range_clamps_and_orders():
    r = Range(1.5, -0.2)
    assert (r.low, r.high) == (0.0, 1.0)


def test_range_closeness():
    r = Range(0.2, 0.4)
    assert r.closeness(0.3) == 1.0
    assert r.closeness(0.5) == pytest.approx(0.8)
    assert r.closeness(1.0) == 0.0


def test_summary_without_filters():
    assert SearchQuery(text="anything").summary() == "no filters (semantic search only)"
