"""Replay StateBench integer-code programs: ground-truth state and token roles.

StateBench prompts are self-contained programs, so everything the analysis
needs about a sample can be recovered from its ``prefix`` alone:

- the register state after every statement (the probe targets for branching
  readout at arbitrary checkpoints);
- per register, three covariates that separate the competing explanations for
  long-length decay: how long ago the value entered the system (``origin``),
  how many reference/increment hops it has been carried through (``hops``),
  and when the register was last written (``last_write``);
- a role for every token (target, arrow, source, value, ...), so gate
  statistics can be split by what the token does rather than pooled.

Statements follow ``state-bench/configs/formats/integer-code*.yaml``::

    x <- 52;                    constant (every class)
    x <- y;                     reference (periodic; TMP is an ordinary register)
    x <- (x + 37) mod 100;      increment (solvable, modular vocabulary)
    // .....                    filler (easy / medium / hard distractors)
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

HEADER_SEPARATOR = "\n\n"
TEMPORARY_VARIABLE = "TMP"

# Alternation order matters: the increment form must be tried before the
# reference form, which would otherwise fail on its parenthesis.
STATEMENT_RE = re.compile(
    r"(?P<increment>(?P<inc_target>[A-Za-z]+) (?P<inc_arrow><-) (?P<lparen>\()(?P<self>[A-Za-z]+)"
    r" (?P<plus>\+) (?P<delta>\d+)(?P<rparen>\)) (?P<mod>mod) (?P<modulus>\d+)(?P<inc_semi>;))"
    r"|(?P<constant>(?P<const_target>[A-Za-z]+) (?P<const_arrow><-) (?P<value>\d+)(?P<const_semi>;))"
    r"|(?P<reference>(?P<ref_target>[A-Za-z]+) (?P<ref_arrow><-) (?P<source>[A-Za-z]+)(?P<ref_semi>;))"
    r"|(?P<filler>// (?:stale log; do not execute: [A-Za-z]+ <- \d+;|unrelated note \d+|\.+))"
)

# Token roles. Codes index ROLE_NAMES; plots group by name.
ROLE_NAMES: tuple[str, ...] = (
    "header",
    "space",
    "filler",
    "target",
    "arrow",
    "value",
    "source",
    "self",
    "lparen",
    "plus",
    "delta",
    "rparen",
    "mod",
    "modulus",
    "semicolon",
)
ROLE_CODES: dict[str, int] = {name: code for code, name in enumerate(ROLE_NAMES)}

# Regex group -> role, per statement kind.
_GROUP_ROLES: dict[str, dict[str, str]] = {
    "increment": {
        "inc_target": "target",
        "inc_arrow": "arrow",
        "lparen": "lparen",
        "self": "self",
        "plus": "plus",
        "delta": "delta",
        "rparen": "rparen",
        "mod": "mod",
        "modulus": "modulus",
        "inc_semi": "semicolon",
    },
    "constant": {
        "const_target": "target",
        "const_arrow": "arrow",
        "value": "value",
        "const_semi": "semicolon",
    },
    "reference": {
        "ref_target": "target",
        "ref_arrow": "arrow",
        "source": "source",
        "ref_semi": "semicolon",
    },
    "filler": {"filler": "filler"},
}


@dataclass(frozen=True)
class Statement:
    kind: str  # "constant" | "reference" | "increment" | "filler"
    target: str | None
    # Constant: the value label. Reference: the source register. Increment: the delta.
    argument: str | None
    modulus: int | None
    start: int  # character offsets into the full prefix
    end: int


@dataclass(frozen=True)
class RegisterState:
    value: str
    # Statement index of the constant write that started this value's chain.
    origin: int
    # Reference copies and increments applied since that constant write.
    hops: int
    # Statement index of the most recent write to this register, of any kind.
    last_write: int


@dataclass
class Program:
    prefix: str
    header_end: int
    statements: list[Statement]

    @property
    def registers(self) -> list[str]:
        """Every register the program writes, in answer order (letters, then TMP)."""
        names = {s.target for s in self.statements if s.target is not None}
        return answer_order(names)

    @property
    def assignment_indices(self) -> np.ndarray:
        """Statement indices of the non-filler statements."""
        return np.array([i for i, s in enumerate(self.statements) if s.kind != "filler"])


def answer_order(names: Iterable[str]) -> list[str]:
    """StateBench lists registers alphabetically with the temporary last."""
    return sorted(names, key=lambda name: (name == TEMPORARY_VARIABLE, name))


def parse_program(prefix: str) -> Program:
    """Parse a StateBench integer-code prompt into statements with character spans."""
    header_end = prefix.find(HEADER_SEPARATOR)
    if header_end < 0:
        raise ValueError("prefix has no task-description separator")
    position = header_end + len(HEADER_SEPARATOR)
    statements: list[Statement] = []
    while position < len(prefix):
        if prefix[position] == " ":
            position += 1
            continue
        match = STATEMENT_RE.match(prefix, position)
        if match is None:
            snippet = prefix[position : position + 60]
            raise ValueError(f"unparseable program text at offset {position}: {snippet!r}")
        kind = match.lastgroup
        if kind == "increment":
            if match["self"] != match["inc_target"]:
                raise ValueError(f"increment reads another register: {match.group(0)!r}")
            statement = Statement(
                "increment", match["inc_target"], match["delta"], int(match["modulus"]),
                match.start(), match.end(),
            )
        elif kind == "constant":
            statement = Statement(
                "constant", match["const_target"], match["value"], None, match.start(), match.end()
            )
        elif kind == "reference":
            statement = Statement(
                "reference", match["ref_target"], match["source"], None, match.start(), match.end()
            )
        else:
            statement = Statement("filler", None, None, None, match.start(), match.end())
        statements.append(statement)
        position = match.end()
    return Program(prefix=prefix, header_end=header_end, statements=statements)


def apply(state: dict[str, RegisterState], statement: Statement, index: int) -> None:
    """Apply one statement to ``state`` in place."""
    if statement.kind == "filler":
        return
    assert statement.target is not None and statement.argument is not None
    if statement.kind == "constant":
        state[statement.target] = RegisterState(statement.argument, index, 0, index)
    elif statement.kind == "reference":
        source = state.get(statement.argument)
        if source is None:
            raise ValueError(f"statement {index} reads undefined register {statement.argument!r}")
        state[statement.target] = RegisterState(source.value, source.origin, source.hops + 1, index)
    else:
        current = state.get(statement.target)
        if current is None:
            raise ValueError(f"statement {index} increments undefined register {statement.target!r}")
        assert statement.modulus is not None
        value = (int(current.value) + int(statement.argument)) % statement.modulus
        state[statement.target] = RegisterState(str(value), current.origin, current.hops + 1, index)


def states_at(program: Program, checkpoints: Sequence[int]) -> list[dict[str, RegisterState]]:
    """Register state after each statement index in ``checkpoints`` (ascending)."""
    if list(checkpoints) != sorted(checkpoints):
        raise ValueError("checkpoints must be ascending")
    state: dict[str, RegisterState] = {}
    snapshots: list[dict[str, RegisterState]] = []
    pending = iter(checkpoints)
    target = next(pending, None)
    for index, statement in enumerate(program.statements):
        if target is None:
            break
        apply(state, statement, index)
        while target == index:
            snapshots.append(dict(state))
            target = next(pending, None)
    if target is not None:
        raise ValueError(f"checkpoint {target} is past the last statement")
    return snapshots


def final_state(program: Program) -> dict[str, RegisterState]:
    return states_at(program, [len(program.statements) - 1])[0]


def first_complete_statement(program: Program) -> int:
    """First statement index after which every register has been written."""
    remaining = set(program.registers)
    for index, statement in enumerate(program.statements):
        remaining.discard(statement.target)
        if not remaining:
            return index
    raise ValueError("program never writes all of its registers")


def render_answer(state: dict[str, RegisterState], registers: Sequence[str]) -> str:
    """The gold completion for ``state``, formatted exactly like the eval choices."""
    return " ".join(f"{name} ?= {state[name].value};" for name in registers)


def character_roles(program: Program) -> np.ndarray:
    """Role code for every character of the prefix (whitespace is ``space``)."""
    roles = np.full(len(program.prefix), ROLE_CODES["space"], dtype=np.int8)
    roles[: program.header_end + len(HEADER_SEPARATOR)] = ROLE_CODES["header"]
    for statement in program.statements:
        match = STATEMENT_RE.match(program.prefix, statement.start)
        assert match is not None and match.lastgroup == statement.kind
        for group, role in _GROUP_ROLES[statement.kind].items():
            start, end = match.span(group)
            roles[start:end] = ROLE_CODES[role]
    return roles


@dataclass
class TokenLayout:
    """Per-token alignment of a tokenized prefix to the program."""

    roles: np.ndarray  # int8 role code per token
    statement: np.ndarray  # int32 statement index per token, -1 for the header
    # Number of prefix tokens up to and including each statement (prefill length
    # for a checkpoint placed right after that statement).
    statement_end_token: np.ndarray


def token_layout(program: Program, offsets: np.ndarray) -> TokenLayout:
    """Align tokenizer ``offsets`` (``(T, 2)`` char spans) with the program.

    A token's role is the role of its first non-blank character, so ``Ġb``
    is a target and the bare ``Ġ`` before a number is a space.
    """
    offsets = np.asarray(offsets, dtype=np.int64)
    char_roles = character_roles(program)
    text = np.frombuffer(program.prefix.encode("utf-32-le"), dtype=np.uint32)
    blank = np.isin(text, [ord(" "), ord("\n"), ord("\t")])
    # next_nonblank[i] = smallest j >= i with a non-blank char (len if none).
    n = len(text)
    idx = np.where(~blank, np.arange(n), n)
    next_nonblank = np.minimum.accumulate(idx[::-1])[::-1]
    next_nonblank = np.append(next_nonblank, n)

    starts, ends = offsets[:, 0], offsets[:, 1]
    first = next_nonblank[np.clip(starts, 0, n)]
    has_char = first < ends
    anchor = np.where(has_char, first, starts)
    roles = np.where(
        has_char, char_roles[np.clip(anchor, 0, n - 1)], ROLE_CODES["space"]
    ).astype(np.int8)
    header_tokens = anchor < program.header_end + len(HEADER_SEPARATOR)
    roles[header_tokens] = ROLE_CODES["header"]

    statement_starts = np.array([s.start for s in program.statements])
    statement_ends = np.array([s.end for s in program.statements])
    statement = np.searchsorted(statement_starts, anchor, side="right") - 1
    statement[header_tokens] = -1

    end_token = np.searchsorted(ends, statement_ends, side="left") + 1
    if not np.array_equal(ends[end_token - 1], statement_ends):
        raise ValueError("a statement boundary falls inside a token")
    return TokenLayout(roles=roles, statement=statement.astype(np.int32), statement_end_token=end_token)


def clean_endpoints(program: Program) -> np.ndarray:
    """Assignment indices after which the program is between swaps.

    A swap is written ``TMP <- x; x <- y; y <- TMP;`` (in the soluble build,
    three increments starting with TMP), and eval programs end on a complete
    one. Reading the state mid-swap asks about a configuration the answer
    format never follows, so a statement is a clean endpoint only if the next
    assignment starts fresh -- a constant write, or any write into TMP -- or it
    is the last assignment.
    """
    assignments = program.assignment_indices
    statements = program.statements
    clean = []
    for position, index in enumerate(assignments):
        if position + 1 == len(assignments):
            clean.append(index)
            continue
        following = statements[assignments[position + 1]]
        if following.kind == "constant" or following.target == TEMPORARY_VARIABLE:
            clean.append(index)
    return np.array(clean, dtype=int)


def log_spaced_checkpoints(program: Program, count: int, *, minimum: int | None = None) -> list[int]:
    """About ``count`` statement indices, log-spaced in assignment count.

    Checkpoints land on clean endpoints (see :func:`clean_endpoints`) at or
    after the first statement that completes the register set, so every
    register has a defined value and no swap is half-applied.
    """
    assignments = program.assignment_indices
    floor = first_complete_statement(program) if minimum is None else minimum
    clean = clean_endpoints(program)
    eligible = clean[clean >= floor]
    if eligible.size == 0:
        return []
    # Position of each eligible statement in assignment order, 1-based.
    ranks = np.searchsorted(assignments, eligible) + 1
    targets = np.unique(np.geomspace(ranks[0], ranks[-1], num=count).round().astype(int))
    chosen = eligible[np.clip(np.searchsorted(ranks, targets), 0, eligible.size - 1)]
    return sorted(set(int(i) for i in chosen) | {int(eligible[-1])})
