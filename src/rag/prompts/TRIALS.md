# Prompt trials

Active: `lookup_v2`, `compare_v2` (`src/rag/config.py`). A trial prompt stays in this
folder after it loses, so its scores can be reproduced: eval provenance records each
prompt's name and sha256. Never edit a prompt file in place; add a new suffix.

## How these scores were produced

`GENERATE_MODEL=gemma3:12b`, 3 runs, active prompts only:

`LOOKUP_PROMPT=lookup_v2 COMPARE_PROMPT=compare_v2 python -m evals.run_answers --runs 3`

Misses and inventions are defined in the README. `lookup_v3`, `compare_v3`, and
`compare_v4` were not scored in this pass.

## Results

| Trial | Prompts | Answer model | Git sha | Run | Misses | Inventions | Clean cases | Cases with problems |
|---|---|---|---|---|---|---|---|---|
| v2 (active) | lookup_v2, compare_v2 | gemma3:12b | a74dbab | 1 | 2 | 0 | 19 | video-game-minutes, foosball-dispute-compare |
| v2 (active) | lookup_v2, compare_v2 | gemma3:12b | a74dbab | 2 | 2 | 0 | 19 | video-game-minutes, foosball-dispute-compare |
| v2 (active) | lookup_v2, compare_v2 | gemma3:12b | a74dbab | 3 | 2 | 0 | 19 | video-game-minutes, foosball-dispute-compare |

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

## Why v2 stayed

These three runs are the measured baseline for the active pair: 2 misses, 0 inventions,
19 clean cases. The active names stay `lookup_v2` and `compare_v2`.
