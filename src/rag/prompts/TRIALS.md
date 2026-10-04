# Prompt trials

Active: `lookup_v5`, `compare_v2` (`src/rag/config.py`). A trial prompt stays in this
folder after it loses, so its scores can be reproduced: eval provenance records each
prompt's name and sha256. Never edit a prompt file in place; add a new suffix.

## How these scores were produced

The first table is the earlier baseline: `GENERATE_MODEL=gemma3:12b`, 3 runs:

`LOOKUP_PROMPT=lookup_v2 COMPARE_PROMPT=compare_v2 python -m evals.run_answers --runs 3`

Misses and inventions are defined in the README. `lookup_v3`, `compare_v3`, and
`compare_v4` were not scored in this pass. The later table is the comparison that
made `lookup_v4` active.

## Results

| Trial | Prompts | Answer model | Git sha | Run | Misses | Inventions | Clean cases | Cases with problems |
|---|---|---|---|---|---|---|---|---|
| v2 | lookup_v2, compare_v2 | gemma3:12b | a74dbab | 1 | 2 | 0 | 19 | video-game-minutes, foosball-dispute-compare |
| v2 | lookup_v2, compare_v2 | gemma3:12b | a74dbab | 2 | 2 | 0 | 19 | video-game-minutes, foosball-dispute-compare |
| v2 | lookup_v2, compare_v2 | gemma3:12b | a74dbab | 3 | 2 | 0 | 19 | video-game-minutes, foosball-dispute-compare |

All 21 questions took the expected route. Both misses were the same on every run:
`video-game-minutes` returned `conflicting` where the case expects `answered`, and
`foosball-dispute-compare` did not say the dispute rule was removed.

## What the unused trial files change

- `lookup_v3`: adds "A passage about a different activity or rule is not a conflict and is
  not part of the answer. Answer only the activity or rule the question asks about, and do
  not mention the other passage."
- `compare_v3`: adds "When a pair has a <missing> side, say that rule was removed or added.
  Do not state it as a current rule. Answer only the rule the question asks about. Do not
  describe a different section as the change."
- `compare_v4`: keeps only the <missing>-side sentences of `compare_v3`.

## Why lookup_v4 replaced lookup_v2

The table above is the earlier baseline, scored before the repeated-claim check and the
refrigerator case (21 questions). Scored again on the current 22 questions, three runs
each, git sha `7cf3959-dirty`, `compare_v2` held fixed:

| Trial | Prompts | Answer model | Run | Misses | Inventions | Clean cases | Cases with problems |
|---|---|---|---|---|---|---|---|
| v2 | lookup_v2, compare_v2 | gemma3:12b | 1–3 | 3 | 0 | 19 | video-game-minutes, hazmat, foosball-dispute-compare |
| v4 (active) | lookup_v4, compare_v2 | gemma3:12b | 1–3 | 2 | 0 | 20 | dog-adoption, foosball-dispute-compare |

All 22 questions took the expected route on every run. `lookup_v4` stopped treating the
foosball limit as a conflict with video-game time, and it stopped repeating the hazmat
rule. It drops the 5-day pet-leave base when the question asks about adopting a dog.
In that comparison, `foosball-dispute-compare` missed “removed” because the model did not
say it. The printed answer now adds that line from the missing pair, and the check
counts the line.

## lookup_v5

`lookup_v5` is the active lookup prompt. It is `lookup_v4` plus one sentence: "When a
rule extends a base amount, state the base too." That is the 5-day pet-leave base
`lookup_v4` dropped on `dog-adoption`. It has not been scored in a three-run eval yet.
