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


def test_object_name_rejects_plurals_when_inflect_is_supplied():
    from circuit_overlap.data import lietzow_object_name

    fake_plurals = {"strawberries", "cows"}
    is_plural = lambda n: n in fake_plurals  # noqa: E731
    assert lietzow_object_name("cow", is_plural=is_plural) == "cow"
    with pytest.raises(ValueError, match="plural"):
        lietzow_object_name("cows", is_plural=is_plural)
    # Without the callable the plural rule is simply not applied.
    assert lietzow_object_name("cows") == "cows"


def test_object_name_rejects_empty():
    from circuit_overlap.data import lietzow_object_name

    with pytest.raises(ValueError):
        lietzow_object_name("   ")
