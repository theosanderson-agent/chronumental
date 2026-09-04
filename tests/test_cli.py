"""Run the command-line tool once per option, on the bundled test data.

The unit tests hold the maths to dense references, which is the right check
for the maths and reaches none of the wiring. A change to what
uncertainty.node_date_intervals returns broke --robust_passes and shipped,
because every check run before pushing exercised a path that did not call it:
the unit tests do not build a model, and the manual end-to-end run used the
defaults. These are slow, and worth it, because a flag that crashes is the
one failure a user cannot work around.

Skipped where jax is missing, since importing __main__ needs the whole
fitting stack. Step counts are tiny -- this asks whether each path runs and
writes what it promises, not whether it converges.
"""
import csv
import os
import subprocess
import sys

import pytest

pytest.importorskip("jax")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "test_data")
TREE = os.path.join(DATA, "divergence_tree.nexus")
DATES = os.path.join(DATA, "ebola.metadata.csv")
# The bundled tree carries substitutions per site, so the genome length has to
# be given or the rate comes out below one mutation a year and the tool -- as
# it should -- refuses.
GENOME_LENGTH = "18958"

CASES = {
    "defaults": [],
    "no intervals": ["--no_confidence_intervals"],
    "conditional intervals": ["--confidence_conditions_on_clock_rate"],
    "clock filter": ["--clock_filter_iqd", "4"],
    "robust passes": ["--robust_passes", "2"],
    "clock filter and robust passes": ["--clock_filter_iqd", "4",
                                       "--robust_passes", "2"],
    "profile clock rate": ["--profile_clock_rate", "3"],
    "profile and robust": ["--profile_clock_rate", "3",
                           "--robust_passes", "2"],
    "given clock": ["--clock", "22.0"],
    "floating clock": ["--floating_clock_rate"],
    "output in years": ["--output_unit", "years"],
}


def run_case(tmp_path, extra):
    dates_out = os.path.join(str(tmp_path), "dates.tsv")
    tree_out = os.path.join(str(tmp_path), "tree.nwk")
    command = [sys.executable, "-m", "chronumental.__main__",
               "--tree", TREE, "--dates", DATES,
               "--dates_out", dates_out, "--tree_out", tree_out,
               "--steps", "30", "--disable_early_stopping",
               "--treat_mutation_units_as_normalised_to_genome_size",
               GENOME_LENGTH, *extra]
    environment = dict(os.environ,
                       PYTHONPATH=os.path.join(os.path.dirname(HERE), "src"))
    done = subprocess.run(command, capture_output=True, text=True,
                          env=environment, timeout=900)
    assert done.returncode == 0, (
        f"exited {done.returncode}\nSTDOUT tail:\n{done.stdout[-1500:]}\n"
        f"STDERR tail:\n{done.stderr[-2000:]}")
    return dates_out, done.stdout


@pytest.mark.parametrize("name", sorted(CASES))
def test_each_option_runs_and_writes_dates(tmp_path, name):
    dates_out, stdout = run_case(tmp_path, CASES[name])
    with open(dates_out) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows, "no dates written"
    assert all(row["predicted_date"] for row in rows)
    # Exiting cleanly is not enough. A failure inside the interval code is
    # caught so it cannot cost someone their fit, which means a broken option
    # looks exactly like a working one apart from two missing columns -- how
    # the --robust_passes tip mismatch survived a first pass of these tests.
    assert "Could not compute confidence intervals" not in stdout
    if "--no_confidence_intervals" not in CASES[name]:
        assert "lower_95" in rows[0], f"intervals missing under {name}"


def test_intervals_are_present_by_default_and_absent_when_asked(tmp_path):
    dates_out, _ = run_case(tmp_path, [])
    header = open(dates_out).readline().strip().split("\t")
    for column in ("lower_95", "upper_95", "date_sd_days"):
        assert column in header, f"{column} missing from {header}"
    dates_out, _ = run_case(tmp_path, ["--no_confidence_intervals"])
    header = open(dates_out).readline().strip().split("\t")
    assert header == ["strain", "predicted_date"]


def test_profiling_reports_a_rate_and_a_root_interval(tmp_path):
    _, stdout = run_case(tmp_path, ["--profile_clock_rate", "3"])
    assert "Profiled clock rate" in stdout
    assert "95%" in stdout


def test_intervals_bracket_the_point_estimate(tmp_path):
    dates_out, _ = run_case(tmp_path, [])
    checked = 0
    with open(dates_out) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if not row["lower_95"] or not row["upper_95"]:
                continue
            assert row["lower_95"] <= row["predicted_date"] <= row["upper_95"], (
                f"{row['strain']}: {row['lower_95']} / "
                f"{row['predicted_date']} / {row['upper_95']}")
            checked += 1
    assert checked > 0


def test_dates_outside_the_calendar_come_back_empty_not_clamped():
    """A bound the calendar cannot reach must not be printed as a date.

    Clamping wrote lassa/l a root of -1430 with an interval of 1-01-02 to
    1-01-02: an interval that does not contain its own point estimate, and
    that reads as a confident answer of year one. An empty cell is the honest
    answer, and date_sd_days still carries the width.
    """
    import pandas as pd
    from chronumental.__main__ import _days_to_dates

    origin = pd.Timestamp("2015-06-01")
    # A few days either way, then something no calendar reaches.
    dates = _days_to_dates(origin, [-10.0, 0.0, 10.0, -1e12, float("nan")])
    assert dates[0] is not None and dates[2] is not None
    assert dates[1].year == 2015
    assert dates[3] is None, "an unreachable date must be empty, not clamped"
    assert dates[4] is None, "a non-finite offset must be empty"


def test_a_deep_root_keeps_an_interval_that_contains_it():
    """pandas reaches years datetime does not, and the bounds must follow it.

    Half a million days before a 2015 origin is year -13, which pandas can
    represent and datetime cannot. Bounding against datetime's calendar is
    what produced the lassa/l interval above.
    """
    import pandas as pd
    from chronumental.__main__ import _days_to_dates

    origin = pd.Timestamp("2015-06-01")
    low, middle, high = _days_to_dates(origin, [-740000.0, -739000.0,
                                                -738000.0])
    assert None not in (low, middle, high)
    assert low < middle < high
    assert middle.year < 0, "expected a date before year zero"
