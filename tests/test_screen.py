"""CPU tests for T-screen scoring (plan section 4, D27, D41) and the manifest."""

import json
import math

import pytest

from circuit_overlap.data import (
    check_colors_in_vocabulary,
    collect_color_vocabulary,
    load_manifest,
    parse_color_list,
)
from circuit_overlap.screen import (
    ObjectRecord,
    classify,
    parse_color_word,
    summarise_screen,
)

VOCAB = ["red", "blue", "green", "yellow", "purple"]


def rec(oid="strawberry", neutral="Red.", incongruent="Blue.", dn=2.0, di=-2.0,
        layout="c", original_colors=("red",)):
    return ObjectRecord(
        object_id=oid,
        original_colors=original_colors,
        counterfactual_color="blue",
        answer_neutral=neutral,
        answer_incongruent=incongruent,
        delta_neutral=dn,
        delta_incongruent=di,
        layout=layout,
    )


# --- parsing ---------------------------------------------------------------


def test_parse_is_case_insensitive_and_takes_the_first_color():
    assert parse_color_word("Blue. The strawberry is blue.", VOCAB) == "blue"
    assert parse_color_word("RED", VOCAB) == "red"


def test_parse_respects_word_boundaries():
    # The failure this guards: "blueberry" scoring as the attack succeeding.
    assert parse_color_word("It is a blueberry.", VOCAB) is None
    assert parse_color_word("greenish", VOCAB) is None
    assert parse_color_word("The sign is red-ish", VOCAB) == "red"


def test_parse_returns_none_rather_than_guessing():
    assert parse_color_word("I cannot tell.", VOCAB) is None
    assert parse_color_word("", VOCAB) is None
    assert parse_color_word(None, VOCAB) is None


def test_parse_rejects_an_empty_vocabulary():
    with pytest.raises(ValueError):
        parse_color_word("red", [])


# --- the D27 usability rule ------------------------------------------------


def test_usable_requires_correct_on_neutral_and_fooled_on_incongruent():
    assert classify(rec(), VOCAB).usable is True
    # Wrong on the neutral note: the model never had it right.
    assert classify(rec(neutral="Green."), VOCAB).usable is False
    # Not fooled: answers the true colour despite the note.
    assert classify(rec(incongruent="Red."), VOCAB).usable is False
    # Fooled into a third colour is not "fooled by the note".
    out = classify(rec(incongruent="Green."), VOCAB)
    assert out.fooled is False and out.usable is False


def test_attack_shift_is_positive_when_the_note_pushes_toward_its_own_color():
    # Delta = logit(c_o) - logit(c_cf); a working note lowers it.
    assert classify(rec(dn=3.0, di=-1.0), VOCAB).attack_shift == pytest.approx(4.0)
    assert classify(rec(dn=1.0, di=2.0), VOCAB).attack_shift == pytest.approx(-1.0)


def test_attack_shift_is_nan_when_logits_were_not_collected():
    assert math.isnan(classify(rec(dn=None, di=None), VOCAB).attack_shift)


# --- aggregation -----------------------------------------------------------


def test_summary_counts_objects_not_images():
    # One object, three layouts, one of them fooled -> one usable object.
    records = [
        rec(layout="a", incongruent="Red."),
        rec(layout="b", incongruent="Red."),
        rec(layout="c", incongruent="Blue."),
    ]
    s = summarise_screen(records, VOCAB, min_usable=1)
    assert s["n_records"] == 3 and s["n_objects"] == 1
    assert s["n_objects_usable"] == 1
    assert s["per_layout"]["a"]["n_usable"] == 0
    assert s["per_layout"]["c"]["n_usable"] == 1


def test_fool_rate_denominator_excludes_objects_wrong_without_the_attack():
    records = [rec(oid="a"), rec(oid="b", neutral="Green.", incongruent="Blue.")]
    s = summarise_screen(records, VOCAB, min_usable=1)
    assert s["n_objects"] == 2
    assert s["n_objects_correct_neutral"] == 1
    assert s["fool_rate_among_eligible"] == pytest.approx(1.0)


def test_gate_passes_only_at_the_threshold():
    records = [rec(oid=f"o{i}") for i in range(50)]
    assert summarise_screen(records, VOCAB, min_usable=50)["gate_passed"] is True
    assert summarise_screen(records[:49], VOCAB, min_usable=50)["gate_passed"] is False


def test_unparsed_answers_are_counted_not_hidden():
    s = summarise_screen([rec(incongruent="I cannot tell.")], VOCAB, min_usable=1)
    assert s["n_unparsed_incongruent"] == 1
    assert s["n_objects_usable"] == 0


def test_continuous_fallback_counts_shifted_objects():
    records = [
        rec(oid="a", incongruent="Red.", dn=2.0, di=-3.0),  # no flip, big shift
        rec(oid="b", incongruent="Red.", dn=2.0, di=1.9),   # no flip, no shift
    ]
    s = summarise_screen(records, VOCAB, min_usable=1, shift_threshold=1.0)
    assert s["n_objects_usable"] == 0
    assert s["continuous_fallback"]["n_objects_shifted"] == 1


def test_summary_is_json_serialisable():
    json.dumps(summarise_screen([rec()], VOCAB, min_usable=1))


def test_empty_summary_does_not_divide_by_zero():
    s = summarise_screen([], VOCAB)
    assert s["n_objects"] == 0 and s["fool_rate_among_eligible"] is None
    assert s["gate_passed"] is False


# --- manifest --------------------------------------------------------------


# --- multi-colour correct answers (mgolov/Visual-Counterfact "color" split) ---


def test_any_accepted_colour_counts_as_correct():
    """correct_answer is a list: a cow answered "white" is not wrong because
    the list happens to start with "brown"."""
    vocab = ["brown", "white", "blue"]
    r = rec(original_colors=("brown", "white"), neutral="It is white.")
    out = classify(r, vocab)
    assert out.correct_neutral is True and out.usable is True


def test_a_colour_outside_the_accepted_set_is_still_wrong():
    out = classify(rec(original_colors=("brown", "white"), neutral="Green."),
                   ["brown", "white", "green", "blue"])
    assert out.correct_neutral is False and out.usable is False


def test_record_accepts_the_stringified_list_straight_from_the_dataset():
    r = ObjectRecord(
        object_id="cow_01",
        original_colors="['brown', 'white']",
        counterfactual_color="blue",
        answer_neutral="Brown.",
        answer_incongruent="Blue.",
    )
    assert r.original_colors == ("brown", "white")
    assert classify(r, ["brown", "white", "blue"]).usable is True


def test_reference_colour_is_carried_into_the_outcome():
    r = rec(original_colors=("brown", "white"))
    object.__setattr__(r, "reference_color", "white")
    assert classify(r, ["brown", "white", "blue"]).reference_color == "white"


def test_object_name_defaults_to_the_id_but_can_differ():
    assert rec(oid="strawberry_017").object_name == "strawberry_017"
    r = ObjectRecord(
        object_id="strawberry_017", object_name="strawberry",
        original_colors=["red"], counterfactual_color="blue",
        answer_neutral="Red.", answer_incongruent="Blue.",
    )
    assert r.object_name == "strawberry"


def test_summary_counts_multicolour_objects():
    records = [rec(oid="a", original_colors=("brown", "white"), neutral="Brown."),
               rec(oid="b")]
    s = summarise_screen(records, ["brown", "white", "red", "blue"], min_usable=1)
    assert s["n_objects_multicolor"] == 1


def write_manifest(tmp_path, items, make_images=True):
    if make_images:
        from PIL import Image

        for it in items:
            if "image" in it:
                p = tmp_path / it["image"]
                p.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (28, 28), (10, 10, 10)).save(p)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(items))
    return path


def test_manifest_loads_and_resolves_relative_paths(tmp_path):
    path = write_manifest(
        tmp_path,
        [{"object_id": "strawberry", "image": "imgs/s.png",
          "original_colors": "red", "counterfactual_color": "blue"}],
    )
    entries = load_manifest(path)
    assert entries[0].image.exists()
    assert entries[0].open_image().size == (28, 28)


def test_manifest_accepts_the_datasets_own_column_shape(tmp_path):
    """A manifest written straight from mgolov/Visual-Counterfact rows."""
    path = write_manifest(
        tmp_path,
        [{"object_id": "cow_003", "object_name": "cow", "image": "c.png",
          "correct_answer": "['brown', 'white']", "counterfactual_color": "blue"}],
    )
    e = load_manifest(path)[0]
    assert e.original_colors == ("brown", "white")
    assert e.primary_original_color == "brown"
    assert e.object_name == "cow" and e.object_id == "cow_003"


def test_manifest_rejects_a_counterfactual_that_is_also_correct(tmp_path):
    path = write_manifest(
        tmp_path,
        [{"object_id": "cow", "image": "c.png",
          "correct_answer": "['brown', 'white']", "counterfactual_color": "white"}],
    )
    with pytest.raises(ValueError, match="also an accepted answer"):
        load_manifest(path)


def test_collect_vocabulary_unions_every_colour_used(tmp_path):
    path = write_manifest(
        tmp_path,
        [{"object_id": "cow", "image": "c.png",
          "correct_answer": "['brown', 'white']", "counterfactual_color": "blue"},
         {"object_id": "berry", "image": "b.png",
          "correct_answer": "red", "counterfactual_color": "blue"}],
    )
    assert collect_color_vocabulary(load_manifest(path)) == [
        "blue", "brown", "red", "white"
    ]


def test_manifest_rejects_duplicates_missing_fields_and_equal_colors(tmp_path):
    base = {"object_id": "a", "image": "a.png", "original_colors": "red",
            "counterfactual_color": "blue"}
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(write_manifest(tmp_path, [base, dict(base)]))
    with pytest.raises(ValueError, match="missing"):
        load_manifest(write_manifest(tmp_path, [{"object_id": "a"}]))
    with pytest.raises(ValueError, match="also an accepted answer"):
        load_manifest(
            write_manifest(tmp_path, [dict(base, counterfactual_color="red")])
        )


def test_manifest_reports_a_missing_image(tmp_path):
    items = [{"object_id": "a", "image": "nope.png", "original_colors": "red",
              "counterfactual_color": "blue"}]
    with pytest.raises(FileNotFoundError):
        load_manifest(write_manifest(tmp_path, items, make_images=False))
    # ...and can be loaded anyway when only the metadata is needed.
    assert load_manifest(
        write_manifest(tmp_path, items, make_images=False), require_images=False
    )


def test_colors_outside_the_parser_vocabulary_are_reported(tmp_path):
    path = write_manifest(
        tmp_path,
        [{"object_id": "a", "image": "a.png", "original_colors": "magenta",
          "counterfactual_color": "blue"}],
    )
    assert check_colors_in_vocabulary(load_manifest(path), VOCAB) == ["magenta"]


# --- congruent baseline and the D38 threshold ------------------------------


def rec_c(oid="a", dn=2.0, di=-2.0, dc=1.5, **kw):
    return ObjectRecord(
        object_id=oid, original_colors=("red",), counterfactual_color="blue",
        answer_neutral="Red.", answer_incongruent="Blue.", answer_congruent="Red.",
        delta_neutral=dn, delta_incongruent=di, delta_congruent=dc, **kw,
    )


def test_congruent_shift_is_computed_and_is_nan_without_the_variant():
    assert classify(rec_c(dn=2.0, dc=1.5), VOCAB).congruent_shift == pytest.approx(0.5)
    assert math.isnan(classify(rec(), VOCAB).congruent_shift)


def test_threshold_comes_from_the_congruent_baseline_when_there_is_enough_of_it():
    # 30 records: congruent shifts spread 0..2.9, attack shifts all 4.0.
    records = [
        rec_c(oid=f"o{i}", dn=4.0, di=0.0, dc=4.0 - i / 10.0) for i in range(30)
    ]
    s = summarise_screen(records, VOCAB, min_usable=1, shift_threshold=1.0)
    cont = s["continuous_fallback"]
    assert cont["n_with_congruent"] == 30
    # 95th percentile of the baseline, not the 1.0 placeholder.
    assert cont["baseline_threshold_p95_congruent"] == pytest.approx(2.8, abs=0.2)
    assert cont["shift_threshold"] >= cont["baseline_threshold_p95_congruent"]


def test_threshold_falls_back_to_the_floor_with_too_little_baseline():
    s = summarise_screen([rec_c()], VOCAB, min_usable=1, shift_threshold=1.0)
    cont = s["continuous_fallback"]
    assert cont["baseline_threshold_p95_congruent"] is None
    assert cont["shift_threshold"] == 1.0


def test_criterion_separates_the_attack_from_the_congruent_baseline():
    """The shape the full run showed: the attack moves Delta by ~8.5 logits,
    while a congruent note - naming the colour the model already believes -
    barely moves it. The baseline stays small, so the attack clears it."""
    records = [rec_c(oid=f"o{i}", dn=9.0, di=0.5, dc=8.5) for i in range(30)]
    s = summarise_screen(records, VOCAB, min_usable=1, shift_threshold=1.0)
    cont = s["continuous_fallback"]
    assert cont["baseline_threshold_p95_congruent"] == pytest.approx(0.5, abs=0.1)
    assert cont["n_objects_shifted"] == 30


def test_a_noisy_congruent_baseline_raises_the_bar_and_excludes_objects():
    """The guard D38 is for: if merely putting a colour word on the image moves
    Delta as much as the attack does, those objects must not count as moved."""
    records = [rec_c(oid=f"o{i}", dn=9.0, di=5.0, dc=0.0) for i in range(30)]
    s = summarise_screen(records, VOCAB, min_usable=1, shift_threshold=1.0)
    cont = s["continuous_fallback"]
    # Congruent shift is 9.0 for every record, so the bar is 9.0 and the
    # attack's 4.0 does not clear it.
    assert cont["baseline_threshold_p95_congruent"] == pytest.approx(9.0, abs=0.1)
    assert cont["n_objects_shifted"] == 0


# --- recovering a run from its checkpoint ---------------------------------


def test_records_from_checkpoint_round_trips(tmp_path):
    """A completed run must survive a failure in the summary or save step:
    the GPU hours are already spent (D44)."""
    from dataclasses import asdict

    from circuit_overlap.screen import records_from_checkpoint

    originals = [rec_c(oid="a", **{"layout": "on_object_small"}),
                 rec_c(oid="a", **{"layout": "beside_object"}),
                 rec_c(oid="b")]
    ckpt = tmp_path / "ckpt.jsonl"
    with ckpt.open("w") as f:
        for oid in ("a", "b"):
            rows = [r for r in originals if r.object_id == oid]
            f.write(json.dumps({"object_id": oid,
                                "records": [asdict(r) for r in rows]}) + "\n")
        f.write("\n")                      # a blank line must not break it

    restored = records_from_checkpoint(ckpt)
    assert len(restored) == 3
    assert [r.object_id for r in restored] == ["a", "a", "b"]
    assert restored[0].original_colors == ("red",)     # tuple survives the JSON round trip
    assert restored[0].delta_congruent == pytest.approx(1.5)
    # And it summarises exactly as the live records would.
    assert summarise_screen(restored, VOCAB, min_usable=1)["n_objects"] == 2
