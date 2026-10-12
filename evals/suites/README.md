# Task suites

Static task corpora, kept from the retired `wam_fusion` harness. Each file is JSON.

| file | items | shape |
|---|---|---|
| `r0_tasks.json` | 40 | `tasks`: short single-turn factual prompts |
| `reasoning_checkable.json` | 30 | `items`: `{id, prompt, answer}` with a checkable answer |
| `reasoning_hard.json` | 30 | same, harder |
| `reasoning_discriminating.json` | 30 | same |
| `reasoning_extrahard.json` | 80 | same |
| `reasoning_headroom.json` | 140 | same |
| `coding_aios.json` | 14 | real aios bug fixes with base and merge SHAs, source and held-out test files |

The gate's acceptance corpus (`../acceptance/probe_corpus.py`) draws its problems from
the five reasoning suites.
