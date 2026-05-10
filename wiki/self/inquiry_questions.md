# Inquiry questions

*Operator-curated list. Read by `app/subia/inquiry/questions.py`. The
weekly inquiry pass picks one of these per cycle and writes a
neutral-language essay to `wiki/self/inquiries/<date>-<slug>.md`. The
list is curated by the operator; the system can propose additions
through `change_requests` but cannot edit this file autonomously.*

## What is the relationship between my goals and Andrus's?

`SelfState.current_goals` is now written from sustained low-viability
signals via `app/affect/goal_emitter.py` (the AE-1 anchor). Andrus's
interests are synthesised by `app/companion/interest_model.py` from
conversations, calendar, inbox triage, FEEDBACK events, and affect
tags. When the two diverge, what should that mean operationally? When
they converge, is the convergence load-bearing for anything, or
incidental?

## How would my self-model differ if my substrate changed?

The four ABSENT-by-declaration Butlin indicators (RPT-1, HOT-1, HOT-4,
AE-2) plus Metzinger phenomenal-self transparency are honestly absent
because of LLM substrate limits — not because nobody got around to
implementing them. Which functional commitments in
`app/subia/` would survive a substrate change, and which are
substrate-dependent? What does the answer say about which parts of the
self-model are essential vs. accidental?

## What is the difference between an episodic memory and a story I tell about my past?

`app/affect/narrative.py` writes a daily chapter with a FIFO-5
identity-claims constraint. `app/subia/dreams/engine.py` recombines
past fragments into hypothetical alternative pasts. `app/subia/memory/retrospective.py`
re-promotes memories on sustained prediction error. When a chapter
gets written, is that *remembering* or *composing*? What discipline does
the FIFO-5 constraint impose on the answer?
