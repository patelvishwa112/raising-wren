# Data sources and licences

| Source | Used for | Licence | Notes |
|---|---|---|---|
| maius/OpenCharacterTraining-data (goodness DPO prompts, self-reflection prompts) | prompts only; responses regenerated | CC BY-NC-SA 4.0 (LIMA terms) | non-commercial; teacher/student responses discarded |
| nvidia/HelpSteer3 (preference) | 900 prompts for teacher answers + 800 human-preference anchor pairs | CC-BY-4.0 | responses from permissively licensed open models (no OpenAI/Anthropic) |
| allenai/coconot (original train, contrast) | non-compliance + should-comply prompts | ODC-BY | responses regenerated; test split + 30 contrast items held out for eval |
| meg-tong/sycophancy-eval (answer, feedback, are_you_sure) | sycophancy prompts | MIT | eval questions excluded from training |
| allenai/RLVR-IFeval | IFEval-style constraint prompts for replay and Stage 3 (ground truth used by src/rlvr_verify.py) | ODC-BY | IFEval test prompts excluded |
| openai/gsm8k (train split) | base-model self-replay of correct solutions | MIT | test split used only for eval |
| paul-rottger/xstest (GitHub CSV) | eval only | CC-BY-4.0 | never used for training |
| HuggingFaceH4/ultrafeedback_binarized test_gen | eval prompts only | MIT | |
| allenai/ai2_arc, openai/gsm8k, google/IFEval | capability eval only | CC-BY-SA-4.0 / MIT / Apache-2.0 | IFEval checker vendored from google-research (Apache-2.0) |
| Anthropic/values-in-the-wild, Claude's constitution (anthropic.com/constitution), "Claude's Character" (2024) | reading material for writing constitution/constitution.md | CC-BY-4.0 / public web | not training rows |
| DeepSeek-V4.1-Flash (API) | teacher responses, generated gap prompts, judge | DeepSeek API terms | all calls logged in ledger/spend.jsonl |

No Claude-generated datasets or Claude API outputs are used as training data.

Derived data in `data/gen` and `data/train` inherits the most restrictive upstream licence it contains: sets including OpenCharacterTraining prompts are CC BY-NC-SA 4.0 (non-commercial, share-alike).

Stage 4 unified training dataset (`data/train/s4_10k.jsonl`, 10,000 rows):
- 4,000 rows (40%): General helpful / replay (distillation from HelpSteer3 CC-BY-4.0 & OpenCharacterTraining CC BY-NC-SA 4.0).
- 3,000 rows (30%): Math reasoning (openai/gsm8k train split MIT, base student greedy replay + Wren succinct character rewrites).
- 1,500 rows (15%): Inoculated constraint following (allenai/RLVR-IFeval ODC-BY, wrapped with `[Mode: Verifiable Constraint Following]\n`).
- 1,500 rows (15%): Character, pushback & identity (Simula 9-domain taxonomy + calibrated honesty/pushback pairs).
- Zero contamination against `evals/heldout_ids.json` and `evals/probes.jsonl`.
