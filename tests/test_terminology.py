"""Tests for the terminology matching & flagging service (Agent 2)."""

from __future__ import annotations

import pytest

from src.services import terminology as term


# ─────────────────────────── LOINC matching ──────────────────────────────────


def test_exact_alias_match_returns_loinc_code():
    match = term.match_to_loinc("Hemoglobin")
    assert match is not None
    assert match["loinc_code"] == "718-7"
    assert match["standard_name"] == "Hemoglobin"
    assert match["match_confidence"] == 100.0


def test_match_reads_parenthesised_abbreviation():
    # The abbreviation inside the parentheses is itself a registered alias.
    match = term.match_to_loinc("Mean Corpuscular Volume (MCV)")
    assert match is not None
    assert match["ontology_key"] == "mcv"


def test_fuzzy_match_tolerates_minor_misspelling():
    match = term.match_to_loinc("Hemglobin")  # missing an 'o'
    assert match is not None
    assert match["ontology_key"] == "hemoglobin"
    assert match["match_confidence"] >= term.FUZZY_MATCH_THRESHOLD


def test_unrelated_name_is_left_uncoded():
    # A wrong LOINC code is worse than no code — nonsense input must not match.
    match = term.match_to_loinc("Xyzzyplasm Wobble Factor Q7")
    assert match is None


def test_empty_name_is_left_uncoded():
    assert term.match_to_loinc("") is None
    assert term.match_to_loinc(None) is None


# Regression: these three compound CBC/differential names used to fuzzy-match
# (score > FUZZY_MATCH_THRESHOLD) onto an unrelated ontology entry that merely
# shares a word ("Platelet", "Hematocrit", "Neutrophil") — silently attaching
# the wrong LOINC code and reference range rather than leaving the row uncoded.
def test_mpv_is_not_confused_with_platelet_count():
    match = term.match_to_loinc("Mean Platelet Volume (MPV)")
    assert match is not None
    assert match["ontology_key"] == "mpv"
    assert match["loinc_code"] == "32623-1"


def test_plateletcrit_is_not_confused_with_packed_cell_volume():
    match = term.match_to_loinc("Platelet Hematocrit")
    assert match is not None
    assert match["ontology_key"] == "plateletcrit"
    assert match["loinc_code"] == "51637-7"


def test_nlr_is_not_confused_with_neutrophils_pct():
    match = term.match_to_loinc("Neutrophil Lymphocyte Ratio (NLR)")
    assert match is not None
    assert match["ontology_key"] == "nlr"
    assert match["loinc_code"] is None  # no single standardised LOINC for this ratio


# ────────────────────────── Reference range parsing ──────────────────────────


@pytest.mark.parametrize(
    "raw,expected_low,expected_high",
    [
        ("13 - 17", 13.0, 17.0),
        ("13-17", 13.0, 17.0),
        ("4.5 – 5.5", 4.5, 5.5),  # en dash
        ("17 - 13", 13.0, 17.0),  # printed backwards — still normalised
    ],
)
def test_two_sided_range_parses_both_bounds(raw, expected_low, expected_high):
    parsed = term._parse_printed_range(raw)
    assert parsed.low == expected_low
    assert parsed.high == expected_high


def test_upper_bound_only_range():
    parsed = term._parse_printed_range("< 200")
    assert parsed.low is None
    assert parsed.high == 200.0


def test_lower_bound_only_range():
    parsed = term._parse_printed_range("> 90")
    assert parsed.low == 90.0
    assert parsed.high is None


def test_qualitative_range_text_is_unparsable():
    assert term._parse_printed_range("Not seen") is None
    assert term._parse_printed_range(None) is None


# ──────────────────────────────── Flagging ───────────────────────────────────


def test_value_inside_two_sided_range_is_green():
    verdict = term.evaluate_flag("15", "13 - 17", None, None)
    assert verdict["flag"] == "GREEN"
    assert verdict["range_source"] == "report"


def test_value_marginally_below_range_is_amber():
    # span 100, low 100 -> margin 10; 95 is within [90, 100) of the low bound.
    verdict = term.evaluate_flag("95", "100 - 200", None, None)
    assert verdict["flag"] == "AMBER"


def test_value_far_below_range_is_red():
    verdict = term.evaluate_flag("50", "100 - 200", None, None)
    assert verdict["flag"] == "RED"


def test_value_marginally_above_range_is_amber():
    verdict = term.evaluate_flag("205", "100 - 200", None, None)
    assert verdict["flag"] == "AMBER"


def test_value_far_above_range_is_red():
    verdict = term.evaluate_flag("260", "100 - 200", None, None)
    assert verdict["flag"] == "RED"


def test_upper_bound_only_flagging():
    assert term.evaluate_flag("180", "< 200", None, None)["flag"] == "GREEN"
    assert term.evaluate_flag("215", "< 200", None, None)["flag"] == "AMBER"  # within 10% margin
    assert term.evaluate_flag("260", "< 200", None, None)["flag"] == "RED"


def test_non_numeric_value_without_qualitative_ontology_is_unknown():
    verdict = term.evaluate_flag("Hazy", "13 - 17", None, None)
    assert verdict["flag"] == "UNKNOWN"


# ──────────────── Printed range vs. ontology default precedence ─────────────


def test_printed_range_overrides_ontology_default():
    match = term.match_to_loinc("Hemoglobin")  # ontology default: 12.0 - 16.0
    # A lab's own (unusually wide) printed interval must win over the ontology.
    verdict = term.evaluate_flag("17.5", "10 - 20", match, None)
    assert verdict["flag"] == "GREEN"
    assert verdict["range_source"] == "report"


def test_ontology_default_used_when_report_prints_no_range():
    match = term.match_to_loinc("Serum Creatinine")
    verdict = term.evaluate_flag("1.0", None, match, "female")  # female default: 0.6 - 1.1
    assert verdict["flag"] == "GREEN"
    assert verdict["range_source"] == "ontology_default"


def test_ontology_default_is_sex_specific():
    match = term.match_to_loinc("Serum Creatinine")
    # 1.2 is inside the male range (0.7-1.3) but outside the female range (0.6-1.1).
    male_verdict = term.evaluate_flag("1.2", None, match, "male")
    female_verdict = term.evaluate_flag("1.2", None, match, "female")
    assert male_verdict["flag"] == "GREEN"
    assert female_verdict["flag"] in ("AMBER", "RED")


# ────────────────────────────── Qualitative path ─────────────────────────────


def test_qualitative_normal_value_is_green():
    match = term.match_to_loinc("Urine Sugar")
    assert match is not None
    verdict = term.evaluate_flag("Nil", None, match, None)
    assert verdict["flag"] == "GREEN"
    assert verdict["range_source"] == "qualitative"


def test_qualitative_abnormal_value_is_red():
    match = term.match_to_loinc("Urine Sugar")
    verdict = term.evaluate_flag("3+", None, match, None)
    assert verdict["flag"] == "RED"


# ─────────────────────────── End-to-end row/report ───────────────────────────


def test_process_item_produces_full_lab_result_shape():
    item = {
        "test_name": "Hemoglobin (Hb)",
        "observed_value": "9.1",
        "unit": "g/dL",
        "reference_range": "13.0 - 17.0",
        "panel": "Complete Blood Count (CBC)",
    }
    row = term.process_item(item, sex="male")

    assert row["test_name"] == "Hemoglobin (Hb)"
    assert row["standard_name"] == "Hemoglobin"
    assert row["loinc_code"] == "718-7"
    assert row["flag"] == "RED"  # 9.1 is well below the printed 13-17 range
    assert row["match_confidence"] is not None
    assert row["reference_range"] == "13.0 - 17.0"  # printed range preserved verbatim


def test_process_item_leaves_unit_and_range_uncoded_gracefully():
    item = {"test_name": "Totally Unknown Marker", "observed_value": "42"}
    row = term.process_item(item)

    assert row["loinc_code"] is None
    assert row["standard_name"] is None
    assert row["flag"] == "UNKNOWN"


def test_annotate_items_passes_patient_sex_through():
    items = [{"test_name": "Serum Creatinine", "observed_value": "1.2"}]
    female_rows = term.annotate_items(items, {"sex": "Female"})
    male_rows = term.annotate_items(items, {"sex": "Male"})

    assert female_rows[0]["flag"] in ("AMBER", "RED")
    assert male_rows[0]["flag"] == "GREEN"


def test_annotate_items_handles_missing_patient_info():
    items = [{"test_name": "Hemoglobin", "observed_value": "14"}]
    rows = term.annotate_items(items, None)
    assert rows[0]["flag"] == "GREEN"
