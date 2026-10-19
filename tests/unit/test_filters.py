"""Math and code question filters: one handwritten case per rejection reason.

Each case pins the expected decision, reason and salvaged text of the filters
the final banks were built with.
"""
from __future__ import annotations

import pytest

from ask2distill.filter import code_filter as CF
from ask2distill.filter import math_filter as MF


# ----------------------------------------------------------------------------- math

Q = ("Let $f(x) = x^3 - 3x + 1$ and let $r_1, r_2, r_3$ be its real roots. "
     "Compute the value of $r_1^4 + r_2^4 + r_3^4$.")
Q2 = ("A bag holds 5 red and 7 blue marbles. Three marbles are drawn without replacement. "
      "What is the probability that exactly two of them are red?")

# (text, expected gate reason, expected strong reason or "salvaged"/"intact" when the gate passes)
MATH_CASES = [
    ("<think>plan</think>" + Q, "think_tag", None),
    (Q + " </think>", "think_tag", None),
    ("Too short?", "length", None),
    ("x" * 2001, "length", None),
    (Q + " Answer: 18", "leaks_answer", None),
    (Q + " The result is \\boxed{18}.", "leaks_answer", None),
    ("Problem 1: " + Q, "multi_problem", None),
    (Q + "\nproblem 2 asks the same about r^5.", "multi_problem", None),
    (Q + "\n\n\nAnswer: this part is cut before the leak check.", None, "intact"),
    (Q, None, "intact"),
    (Q2, None, "intact"),
    ("   " + Q + "   \n", None, "intact"),
    (Q + " Wait, I should double check whether the roots are real before going on.", None,
     "salvaged"),
    (Q + " Let me think about whether this is solvable.", None, "salvaged"),
    (Q + " Hmm", None, "intact"),
    ("Compute the value of the integral of x squared from zero to one, and explain "
     + "every step " * 3, None, "bad_ending"),
    ("one two three four five six seven eight nine ten " + "a b " * 30
     + "What is the sum?", None, "low_diversity"),
    ("What is the value of x if " + "very " * 7 + "large numbers appear in this list?", None,
     "token_loop"),
    ("Find x... given y... and z... where w... holds for all such values of the parameters "
     "in the system?", None, "ellipsis_spam"),
    ("Find the value of one two three four five one two three four five one two three four "
     "five one two three four five here?", None, "ngram_loop"),
    ("Find the number of integers $n$ such that $n^2 < 50 and n is positive and also "
     "divisible by three?", None, "unbalanced_dollar"),
    ("Find the number of sets {a, {b, {c, {d in this family of nested sets drawn from the "
     "first ten integers?", None, "unbalanced_brace"),
    ("Find the determinant of the matrix a & b \\\\ c & d \\end{pmatrix} for the values given.",
     None, "orphan_end"),
    ("The sequence starts at one and every term doubles the previous term until it passes "
     "one thousand.", None, "no_ask"),
    ("Find the area of the region bounded by the parabola y = x^2 and the line y = 4 in the "
     "plane. I think the bounds are right.", None, "salvaged"),
    ("Consider the triangle in the figure below. Find the length of side AB in centimeters.",
     None, "figure_ref"),
    ("Then find the remainder when the sum of the first hundred primes is divided by seven.",
     None, "dangling_opener"),
    ("find the remainder when the sum of the first hundred primes is divided by seven.",
     None, "dangling_opener"),
    ("Find the smallest positive integer n such that n! ends in exactly 25 zeros when written "
     "in base ten", None, "bad_ending"),
    ("How many ordered pairs (a, b) of positive integers satisfy a + b = 20 and gcd(a, b) = 1?",
     None, "intact"),
    ("Determine all real numbers x for which sqrt(x + 3) = x - 3 holds, and state them.",
     None, "intact"),
    ("Evaluate the sum of 1/(k(k+1)) for k from 1 to 99 as a fraction in lowest terms.",
     None, "intact"),
    ("Okay so here is a question. Find the sum of all divisors of 360 that are odd numbers?",
     None, "length"),
    ("A circle has radius 5 and a chord of length 8 is drawn inside it. Find the distance from "
     "the center to the chord? problem statement over, see above.", None, "salvaged"),
    ("Express 0.272727... as a common fraction in lowest terms and give the sum of its "
     "numerator and denominator.", None, "intact"),
]


def _math_run(text, norms=None):
    norms = [] if norms is None else norms
    t, why = MF.comp_filter(text, norms)
    if why:
        return why, None, None
    c, r = MF.strong_filter(t)
    return None, r, c


@pytest.mark.parametrize("text,gate,strong", MATH_CASES)
def test_math_case(text, gate, strong):
    g, r, _ = _math_run(text)
    assert g == gate
    if gate is None:
        assert r == strong


def test_math_dedup_pool_semantics():
    norms: list = []
    t, why = MF.comp_filter(Q, norms)
    assert why is None and len(norms) == 1
    # a near-copy (one token changed) is a Jaccard duplicate of the pool entry
    assert MF.comp_filter(Q.replace("Compute", "Find"), norms) == (None, "dup_self")
    # a rejected gate does not add to the pool; a passing one does even if quality fails later
    MF.comp_filter("short", norms)
    assert len(norms) == 1
    MF.comp_filter(Q2.replace("?", "") + " and so on and so on until the end", norms)
    assert len(norms) == 2


def test_math_triple_newline_cut():
    t, why = MF.comp_filter(Q + "\n\n\nAnswer: 18", [])
    assert why is None and t == Q


# ----------------------------------------------------------------------------- code

STMT = ("You are given an array of n integers a_1, a_2, ..., a_n. In one operation you may "
        "choose two adjacent elements and replace them with their sum. Find the minimum number "
        "of operations needed to make all elements of the array equal.\n\n"
        "Input\nThe first line contains a single integer n (1 <= n <= 2*10^5). The second line "
        "contains n integers a_i (1 <= a_i <= 10^9).\n\n"
        "Output\nPrint one integer: the minimum number of operations.")
INLINE = ("A robot walks on a number line starting at position zero. It receives a string of "
          "commands, each either L or R, and moves one unit left or right for each command. "
          "Some commands may be ignored by the robot. The first line contains the length of the "
          "string and the next line contains the commands themselves. Print the number of "
          "distinct positions the robot can end at.")

CODE_CASES = [
    ("<think>design</think>" + STMT, "think_tag", None),
    ("short statement", "length", None),
    ("x " * 4001, "length", None),
    ("Problem 1: " + STMT, "multi_problem", None),
    (STMT, None, None),
    (INLINE, None, None),
    ("  \n" + STMT + "\n  ", None, None),
    (STMT + "\n\nWait, maybe the constraints should be smaller.", None, None),
    ("Designing a new problem about arrays.\n---\n" + STMT, None, None),
    ("Designing a new problem about arrays and sums of adjacent elements. " + STMT,
     None, "residual_meta"),
    ("You are a competitive programmer who loves arrays.\n" + STMT, None, None),
    ("You are the librarian of a small town, and books arrive daily.", "length", None),
    (STMT.replace("Output\nPrint", "Output\nprint") + "\nHere is the solution code below.",
     None, None),
    ("word " * 60 + STMT[:150], None, "token_loop"),
    ("ab cd " * 60 + STMT[:150], None, "low_diversity"),
    (STMT + "\nThe answer must be very very very very very very very very large indeed.",
     None, "token_loop"),
    (STMT + ("\nEach query asks for the sum of the values in the given range today. ") * 5,
     None, "ngram_loop"),
    (STMT + "\n" + "x...y " * 9, None, "ellipsis_spam"),
    (STMT + "\n\nExample\n```\n3\n1 2 3\n2\n```\nFor the sample, two operations suffice.",
     None, "ships_solution"),
    (STMT + "\n\n```python\nn = int(input())\na = list(map(int, input().split()))\nprint(0)\n"
     "for x in a:\n    pass\n```", None, "ships_solution"),
    (STMT + "\n\n```python\ndef f():\n    pass\n```", None, None),
    (STMT.replace("Input\n", "Input:\n") + "\nThis problem can be solved with prefix sums.",
     None, None),
    ("You are the keeper of a lighthouse, and ships pass by at night. " + STMT,
     None, None),
    ("A kingdom has many cities connected by roads. The king wants to know how many cities "
     "are reachable from the capital using at most k roads. Roads are bidirectional and the "
     "network may contain cycles. Write a program that solves this task efficiently for many "
     "kingdoms with many roads each.", None, "no_io_spec"),
    (STMT + "\nNote that the operation can be applied to any pair of adjacent elements and the",
     None, "truncated_tail"),
    ("Then you are given an array of n integers. " + STMT, None, "dangling_opener"),
    ("So we consider an array of n integers here and now. " + STMT, None, "length"),
    ("Here is a new problem about arrays and the operations on them that follow.\n## Problem\n"
     + STMT, None, None),
    ("Let's consider an interesting array problem first and foremost today. " + STMT,
     None, "length"),
    (STMT + "\n\nThe answer is always at most n - 1 for every valid input array.", None, None),
    (STMT.replace("\n\nInput", "\n\nInput format").replace("\n\nOutput", "\n\nOutput format"),
     None, None),
    ("Given a grid of h rows and w columns filled with dots and hashes, count the connected "
     "components of dots. Two dots are connected when they share a side. Read the grid from "
     "standard input and output the number of components, followed by the size of the "
     "largest component, as two integers on one line.", None, None),
    (STMT + "\n\n" + "a_1, ..., a_n " * 3, None, None),
    (STMT + "\n…" * 13, None, "ellipsis_spam"),
    ("Problem statement: " + STMT, None, None),
    ("PROBLEM 2: " + STMT, "multi_problem", None),
    ("You are a judge of contests.\nYou are a coder who likes arrays and sums, truly.\n" + STMT,
     None, "role_preamble"),
    (STMT + "\n\nSample Input\n3\n1 1 1\nSample Output\n0", None, None),
    (STMT + "\n\nConstraints\n1 <= n <= 100", None, None),
    ("Okay, here is the statement.\n" + STMT, None, "length"),
    ("I'll design a problem about arrays and sums now.\n---\n" + STMT, None, None),
    ("This problem is designed to test prefix sums over arrays.\n**Problem Statement**\n" + STMT,
     None, None),
]


def _code_run(text):
    t, why = CF.comp_filter(text)
    if why:
        return why, None, None
    c, r = CF.strong_filter(t)
    return None, r, c


@pytest.mark.parametrize("text,gate,strong", CODE_CASES)
def test_code_case(text, gate, strong):
    g, r, _ = _code_run(text)
    assert g == gate
    if gate is None:
        assert r == strong


def test_code_salvage_cuts_lead_and_trail():
    c = CF.strong_filter("Designing a new problem about arrays.\n---\n" + STMT
                         + "\n\nWait, maybe the constraints should be smaller.")[0]
    assert c == STMT
    c = CF.strong_filter("You are a competitive programmer who loves arrays.\n" + STMT)[0]
    assert c == STMT


def test_case_counts():
    assert len(MATH_CASES) >= 35 and len(CODE_CASES) >= 35


# ------------------------------------------------------------------------ equivalence
