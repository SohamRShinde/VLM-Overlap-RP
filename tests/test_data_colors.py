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
