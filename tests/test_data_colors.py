"""parse_color_list: the Visual-Counterfact 'correct_answer' column is a
stringified Python list, so every entry point has to normalise it."""

import pytest

from circuit_overlap.data import parse_color_list


def test_stringified_list_from_the_dataset():
    assert parse_color_list("['brown', 'white']") == ("brown", "white")
    assert parse_color_list('["brown", "white"]') == ("brown", "white")
    assert parse_color_list("['red']") == ("red",)


def test_bare_string_and_real_sequences():
    assert parse_color_list("red") == ("red",)
    assert parse_color_list(["red", "blue"]) == ("red", "blue")
    assert parse_color_list(("red",)) == ("red",)


def test_normalises_case_and_whitespace():
    assert parse_color_list("  Red ") == ("red",)
    assert parse_color_list("['Brown', ' WHITE ']") == ("brown", "white")


def test_preserves_order_and_drops_duplicates():
    # Order matters: the first element is the fallback reference colour.
    assert parse_color_list("['white', 'brown', 'white']") == ("white", "brown")


def test_rejects_empty_and_unparseable_fields():
    for bad in ("", "   ", "[]", "['', '  ']"):
        with pytest.raises(ValueError):
            parse_color_list(bad)
    with pytest.raises(ValueError):
        parse_color_list("['unclosed'")
    with pytest.raises(ValueError):
        parse_color_list(None)


def test_rejects_non_string_entries_and_types():
    with pytest.raises(TypeError):
        parse_color_list([1, 2])
    with pytest.raises(TypeError):
        parse_color_list(3.5)


def test_does_not_execute_code_in_the_field():
    # literal_eval only: a call expression must fail, not run.
    with pytest.raises(ValueError):
        parse_color_list("[__import__('os').getcwd()]")


# --- Lietzow's own normalisation (repo read 2026-10-19) --------------------


def test_grey_and_gray_are_one_colour():
    """Their _sanitize_color_name folds both spellings; splitting the logit
    mass across two tokens would understate the model's confidence."""
    assert parse_color_list("grey") == ("gray",)
    assert parse_color_list("['Grey', 'gray']") == ("gray",)


def test_object_name_rejects_parenthetical_extra_information():
    from circuit_overlap.data import lietzow_object_name

    assert lietzow_object_name("Strawberry") == "strawberry"
    assert lietzow_object_name("fire_truck") == "fire truck"
    with pytest.raises(ValueError, match="extra information"):
        lietzow_object_name("apple (fruit)")


def test_plural_test_receives_the_capitalized_name():
    """Their _sanitize_object_name capitalizes before the plural check."""
    from circuit_overlap.data import lietzow_object_name

    seen = []
    lietzow_object_name("fire_truck", is_plural=lambda n: seen.append(n) or False)
    assert seen == ["Fire truck"]


def test_plural_rule_keeps_words_inflect_returns_unchanged():
    """deer, sheep, moose, goldfish, mackerel, 'anemone fish': singular_noun
    returns the word itself, so Lietzow keep them. Checking truthiness alone
    would drop all six and cost 6 objects against their 469."""
    from circuit_overlap.data import lietzow_object_name

    class FakeEngine:
        # Mirrors inflect 7.5.0 on the real data (observed 2026-10-19).
        table = {"Deer": "Deer", "Sheep": "Sheep", "Beans": "Bean", "Bus": "Bu"}

        def singular_noun(self, word):
            return self.table.get(word, False)

    eng = FakeEngine()
    assert lietzow_object_name("deer", inflect_engine=eng) == "deer"
    assert lietzow_object_name("sheep", inflect_engine=eng) == "sheep"
    for plural in ("beans", "bus"):
        with pytest.raises(ValueError, match="plural"):
            lietzow_object_name(plural, inflect_engine=eng)


def test_plural_rule_is_skipped_without_an_engine():
    from circuit_overlap.data import lietzow_object_name

    assert lietzow_object_name("cows") == "cows"


def test_object_name_rejects_empty():
    from circuit_overlap.data import lietzow_object_name

    with pytest.raises(ValueError):
        lietzow_object_name("   ")



def test_require_list_rejects_a_bare_string_as_lietzow_do():
    """correct_answer is typed list[str]; a bare string fails their validation
    and the row disappears. Suspected cause of the last 2 objects of the
    493 -> 469 -> 467 cascade."""
    assert parse_color_list("red", require_list=False) == ("red",)
    with pytest.raises(ValueError, match="bare string"):
        parse_color_list("red", require_list=True)
    # A stringified list is still fine under the strict rule.
    assert parse_color_list("['red']", require_list=True) == ("red",)


def test_a_stringified_non_list_is_rejected():
    with pytest.raises(ValueError, match="not a list"):
        parse_color_list("{'red': 1}")
