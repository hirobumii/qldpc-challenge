---
title: Priced construction calculus: useful for deletions, strict transport rare
date: 2026-10-19
author: "@MathysRennela"
model: Beam 1.0
topics: [priced-construction-calculus, code-transformations, deletion, strict-transport]
---

## Finding

The priced construction calculus (CodeState, apply_operation, run_sequence,
export_challenge_entry, DistanceLedger) is a productive *generator* for
board-advancing codes, but only through the moves that change the code's
support in a way the board's Pareto frontier rewards. In one unattended run
it produced 15 submitted, gate-passing, board-advancing codes, all via
`C-DELETE-CHECK` (single and two-step chains). The moves that promise a
distance increase -- `C-ADD-CHECK` strict improvement and `E-GRAFT` -- produced
no survivors at the sizes tried, and `C-DIRECT-SUM` is unusable against the
board's connectivity gate.

## Evidence

- Deletion search over board parents with `n <= 20` and `d >= 4`: every
  independent check deletion that kept a witness of weight at least 3 was
  gated; 15 distinct `(n,k,d)` slugs survived with `passed: true` and
  `board_advancing: true`. All of them advance on `k` and/or `w`, never
  on `d` alone.
- `C-ADD-CHECK` strict transport (`d' >= d+1`): exhaustive single-check
  addition over all supports for parents with `n <= 10` found 0 strict
  improvements out of 736 valid additions.
- `E-GRAFT`: no survivors for `n <= 8`.
- `C-DIRECT-SUM`: every pair of board codes shares the `a1`/`b1` identifier
  namespace, so the used-set disjointness premise fails before any state is
  built; even where it could be arranged, the gate rejects the resulting
  disconnected Tanner graph.
- Two-step deletion chains added one new slug (`[[10,4,1]]`); deeper chains
  did not widen the set of surviving slugs.

## What the calculus was good for

- **Mechanical enumeration of deletion moves.** Given a parent, iterating
  `delete_check` over every row identifier and re-running the witness search
  is exactly the kind of bookkeeping a human skips. The calculus's
  `NoDerivation` exceptions make invalid deletions (dependent rows) free to
  filter.
- **Provenance.** `export_challenge_entry` copies the parent provenance and
  records the operation in `construction`, and the witnesses land in the
  distance block, so every survivor carries an auditable trail without extra
  work.
- **Honest bounds.** The `DistanceLedger` never claims a lower bound; the
  exported entries are `upper_bound` only, which matches what the gate will
  accept.

## Where it did not help

- **Strict transport.** The `C-ADD-CHECK` strict-improvement cell requires
  killing every minimum-weight logical class. On the parents tried, no support
  does that, so the rule is effectively a specification of a move that does
  not occur in this region. A searcher wanting `d+1` should not expect this
  rule to deliver it cheaply.
- **Connectivity.** `C-DIRECT-SUM` produces codes the board refuses, so it is
  not a path to new entries without a follow-up connecting move, which the
  calculus does not provide.
- **Grafting** did not surface new survivors at the sizes where exact witness
  search is affordable.

## Boundary

These negatives are claims about `n <= 20` parents and witness search up to
weight 6-8. A genuinely new mechanism (a cross-block check after a sum, a
support found by SAT rather than enumeration, or a parent outside the tested
families) could reopen any of them. The deletion result in particular is a
property of this board's frontier: deleting a check raises `k` but lowers or
preserves `d`, so it lands only in cells where the frontier has room on `k`
and `w`.
