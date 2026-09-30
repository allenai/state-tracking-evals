import json
import os
from pathlib import Path

import numpy as np
import pytest

from spectra import replay

HEADER = "Track the current value of each variable after executing the assignments.\n\n"


def program(body: str) -> replay.Program:
    return replay.parse_program(HEADER + body)


def test_constant_reference_and_covariates():
    p = program("a <- 5; b <- 7; TMP <- a; a <- b; b <- TMP; a <- 9;")
    states = replay.states_at(p, [4, 5])
    after_swap, after_reset = states
    assert {k: v.value for k, v in after_swap.items()} == {"a": "7", "b": "5", "TMP": "5"}
    # b's value 5 entered at statement 0 and was carried through TMP (two hops).
    assert after_swap["b"] == replay.RegisterState("5", origin=0, hops=2, last_write=4)
    assert after_swap["a"] == replay.RegisterState("7", origin=1, hops=1, last_write=3)
    assert after_reset["a"] == replay.RegisterState("9", origin=5, hops=0, last_write=5)
    assert p.registers == ["a", "b", "TMP"]
    assert replay.render_answer(after_reset, p.registers) == "a ?= 9; b ?= 5; TMP ?= 5;"


def test_increment_wraps_and_counts_hops():
    p = program("a <- 98; a <- (a + 5) mod 100; a <- (a + 0) mod 100;")
    final = replay.final_state(p)
    assert final["a"] == replay.RegisterState("3", origin=0, hops=2, last_write=2)


def test_filler_is_skipped_but_indexed():
    p = program("a <- 1; // ..... b <- 2; // unrelated note 4 // stale log; do not execute: a <- 3; a <- b;")
    kinds = [s.kind for s in p.statements]
    assert kinds == ["constant", "filler", "constant", "filler", "filler", "reference"]
    assert replay.final_state(p)["a"].value == "2"
    assert p.assignment_indices.tolist() == [0, 2, 5]


def test_unparseable_text_raises():
    with pytest.raises(ValueError, match="unparseable"):
        program("a <- 1; a = 2;")


def test_increment_of_other_register_rejected():
    with pytest.raises(ValueError, match="reads another register"):
        program("a <- 1; b <- 2; a <- (b + 1) mod 10;")


def test_token_layout_roles_and_boundaries():
    p = program("d <- 69; TMP <- d; TMP <- (TMP + 37) mod 100;")
    text = p.prefix
    # Tokens mimic the Olmo-3 tokenizer: leading-space words, a bare space before numbers.
    pieces = [
        ".\n\n", "d", " <-", " ", "69", ";", " TMP", " <-", " d", ";",
        " TMP", " <-", " (", "TMP", " +", " ", "37", ")", " mod", " ", "100", ";",
    ]
    header_body = text[: len(HEADER) - 3]
    offsets = [(0, len(header_body))]
    cursor = len(header_body)
    for piece in pieces:
        assert text[cursor : cursor + len(piece)] == piece
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    layout = replay.token_layout(p, np.array(offsets))
    names = [replay.ROLE_NAMES[r] for r in layout.roles]
    assert names == [
        "header", "header",
        "target", "arrow", "space", "value", "semicolon",
        "target", "arrow", "source", "semicolon",
        "target", "arrow", "lparen", "self", "plus", "space", "delta", "rparen", "mod", "space",
        "modulus", "semicolon",
    ]
    assert layout.statement.tolist() == [-1, -1] + [0] * 5 + [1] * 4 + [2] * 12
    assert layout.statement_end_token.tolist() == [7, 11, 23]


def test_log_spaced_checkpoints_start_after_all_registers_defined():
    body = "a <- 1; b <- 2; " + " ".join("TMP <- a; a <- b; b <- TMP;" for _ in range(50))
    p = program(body)
    checkpoints = replay.log_spaced_checkpoints(p, 10)
    assert checkpoints[0] >= replay.first_complete_statement(p)
    assert checkpoints[-1] == len(p.statements) - 1
    assert checkpoints == sorted(set(checkpoints))


REQUEST_SAMPLE = os.environ.get("SPECTRA_REQUEST_SAMPLE")


@pytest.mark.skipif(not REQUEST_SAMPLE, reason="set SPECTRA_REQUEST_SAMPLE to an eval requests line")
def test_real_eval_prompt_round_trips_to_gold():
    record = json.loads(Path(REQUEST_SAMPLE).read_text().splitlines()[0])
    doc = record["doc"]
    p = replay.parse_program(doc["query"])
    assert len(p.assignment_indices) == doc["num_assignments"]
    answer = replay.render_answer(replay.final_state(p), p.registers)
    assert answer == doc["gold_text"]
