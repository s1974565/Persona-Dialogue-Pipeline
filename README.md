# Persona dialogue simulation pipeline

Code, inputs and run records for the pipeline in the paper, instantiated on Korean older adults living
alone. Everything ran in Korean.

`run.py` is the entry point; `--arm` selects the run. `lca.py` holds the latent class analysis and the
seed sampling, `conversation.py` the dialogue system, `evaluation.py` the judge call, `restoration.py`
the in-character survey, `configurations.py` the settings and record schemas, `resources.py` the loaders
for `data/`, `persistence.py` the JSONL records, `clients.py` the client resolution, `service/` the
provider clients.

`data/` holds the thirty backgrounds, the scenarios, the partner prompt and the judge items; `lca.py`
writes the seed sets there. `prompts/` holds the background and scenario prompts, one judge prompt, and
the 18 items as administered. `results/` holds one JSON record per conversation, per judge call and
per in-character survey.

| directory | conversations | judge records | design |
|---|---|---|---|
| `restoration/` | — | — | 240 surveys: 8 simulators × 30 personas answering the 11 indicators in character |
| `fixed_15turn/` | 1,200 | 12,960 | 8 simulators × 30 personas × 5 scenarios, 15 turns each |
| `free_turn/` | 1,440 | — | 8 × 30 × 6, persona may end the conversation |
| `ablation/background_minimal/` | 450 | 4,860 | 3 simulators, generated background, minimal system |
| `ablation/seed_minimal/` | 450 | 4,860 | 3 simulators, seed conditions as background, minimal system |

The third ablation setting is `fixed_15turn/` restricted to the same three simulators. Conversation records
carry the persona's inner state and its candidate utterances; the two minimal arms carry the spoken
utterance only.

```
pip install -r requirements.txt
cp .env.example .env
python lca.py
python run.py all --arm fixed_15turn
python run.py conversations --arm free_turn
python run.py all --arm background_minimal
python run.py all --arm seed_minimal
python run.py restore
```

Records are appended under `results/`; a run resumes from what is already there.

Expert-study materials are not included.
