# Part 5: Integrated Waste Management Assistant

**Notebook:** [`part5_integrated_assistant.ipynb`](part5_integrated_assistant.ipynb)

## What this notebook does

Joins the three models into one assistant. It accepts a photo, a written description, or both, and returns the waste category, recycling instructions, and the policy text they came from. It measures how reliable the CNN's confidence is, handles bad input and disagreements, tests the whole system, and adds a feedback mechanism.

## Needs

| Item | Notes |
|---|---|
| `waste_cnn.keras` | Created by Part 2 |
| `waste_rag_model/` | Created by Part 4 |
| `RealWaste/` | For the held-out images (not in the repository) |
| `waste_descriptions.csv`, `waste_policy_documents.json` | The source files |
| TensorFlow and PyTorch | Both are loaded in one session (imported torch-first, the order that works) |

Run Parts 2 and 4 first, or copy their saved outputs next to this notebook.

## Creates

Nothing on disk (temporary test images are deleted automatically).

## Libraries

TensorFlow / Keras, PyTorch, transformers, sentence-transformers, scikit-learn, pandas, numpy, Pillow.

## Steps

| Step | What it does |
|---|---|
| 1 to 3 | Libraries; load the saved models; the 950 held-out images (read from their paths) |
| 4 to 7 | Rebuild the corpus, retrieval, generator functions with the checker, and the text classifier |
| 8 | Smoke test of the three models |
| 9 | How reliable is the CNN's confidence? (accuracy by confidence band; non-waste images) |
| 10 | The assistant: input checks, decisions, conflict handling |
| 11 | Edge cases, speed, real test images |
| 12 | End-to-end test: 360 requests |
| 13 | Feedback and learning from corrections |

## Key findings

- **CNN confidence (950 held-out images):** accuracy rises from about 0.54 below 0.7 confidence to 0.69 at 0.7 to 0.8 and **0.93 at 0.8 and above** (62% of images). Cut-offs of **0.80 and 0.70** were chosen where accuracy steps up.
- **Non-waste pictures:** black, white, noise and a gradient get a wrong category at confidence 0.28 to 0.57, so the CNN cannot reject them on its own; at the 0.70 cut-off all four ask for more information.
- **Decisions:** one usable input gives an answer; agreeing inputs give an answer; disagreeing inputs give a **conflict** (the user confirms, instructions for both are shown); nothing usable gives "need more information".
- **Edge cases:** no input, whitespace, a missing file, a wrong file type, gibberish, a Swahili phrase and a non-text input all give a clean message; nothing crashed.
- **Speed:** warming the answer cache takes 32 seconds for nine answers; after that about 0.12 to 0.15 s per image request and about 0.01 s per text request.
- **End-to-end (360 requests):** image only answered 74 of 90 at 0.905 correct; with an agreeing description 83 answered, all correct, and 7 conflicts equal to the 7 wrong image-only answers (counts match; images were not compared one by one); a conflicting description was flagged for 73 of the 74 requests with a usable photo.
- **Feedback:** one correction made a refused Swahili phrase answerable without changing other answers. The guard cannot catch a wrong correction, so corrections need human review.

## Run notes

- Step 9 reads all 950 held-out images (about 30 seconds). Step 11 warms the cache by generating nine answers (about 30 seconds).
- Both models stay in memory; close other notebooks first.
- A CPU is enough. First runs download the MiniLM model.
- If you re-run an earlier cell that defines `assist`, re-run the later ones.

## Limitations

- The CNN cannot recognise non-waste pictures (only synthetic ones tested).
- The text classifier and generator are tested only on template-style data.
- When a photo is unsure, a description is trusted without a check.
- The feedback guard cannot detect a wrong correction, and the log is not saved.
- The held-out images are partly the validation set of Part 2, so image results are slightly optimistic. Each experiment is a single run on a small sample.

## Related notebooks

Loads the CNN from Part 2 and the generator from Part 4, and rebuilds the text model from Part 3.
