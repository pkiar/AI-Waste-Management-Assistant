# Part 3: Waste Description Classification

**Notebook:** [`part3_text_classifier_simple.ipynb`](part3_text_classifier_simple.ipynb)

## What this notebook does

Classifies a short waste description (for example "empty green glass bottle with contents") into one of 9 categories using scikit-learn. It compares simple models, studies the mistakes, adds word embeddings, tests the final model once, and wraps it in a prediction function.

## Needs

| Item | Notes |
|---|---|
| `waste_descriptions.csv` | 5000 waste descriptions |

## Creates

Nothing on disk.

## Libraries

scikit-learn, pandas, numpy, matplotlib, TensorFlow / Keras (for the embedding step).

## Steps

| Step | What it does |
|---|---|
| 2 to 4 | Load, clean, remove duplicates, labels, 80/10/10 stratified split |
| 5 | Baseline models (dummy, Naive Bayes, logistic regression, random forest) on validation |
| 6 | Look at the mistakes of the two best |
| 7 | Why decoy words win (word counts) |
| 8 | Cross-validation of seven model and feature variants |
| 9 | The two best on the validation set |
| 10 | Stress test on unusual descriptions, with a calibrated model |
| 11 | Word embeddings: a Keras embedding model, nearest words, embeddings as features |
| 12 | Final model: one test evaluation |
| 13 | Prediction function with guards for empty and unknown input |

## Key findings

- **Data:** 5000 rows become 4939 after removing duplicates; 3951 train, 494 validation, 494 test.
- **Baselines (validation):** dummy 0.119, Naive Bayes 0.990, logistic regression 0.990, random forest 0.972. Near-perfect scores reflect template text, not real-world accuracy.
- **What the mistakes taught:** in 9 of 10 mistakes a frequent material word (glass, paper, plastic) sits in front of a rare item word (pen, marker, battery); Miscellaneous Trash is the true class in 8 of 10.
- **Cross-validation:** LinearSVC 0.9998, logistic regression with C=10 0.9984, ComplementNB 0.9966. Raising C from 1 to 10 lifted logistic regression from 0.9928 to 0.9984, so regularisation had been suppressing rare item words.
- **Stress test:** 13 of 14 answerable inputs right (all 5 decoys). Unknown text (empty, gibberish, Swahili) gets confidence 0.25, below every correct answer, so a threshold can reject it.
- **Embeddings:** 307-word vocabulary, validation accuracy 1.0. Neighbours group by category, not meaning, because the embeddings are learned from the labels. An empty description gives `nan`, so the final function handles empty input itself.
- **Final model:** TF-IDF (words and pairs) with a calibrated LinearSVC. **Test accuracy 1.0 with 0 errors** on 494 descriptions.

## Run notes

- Steps 1 to 10 and 12 to 13 use only scikit-learn and run in under a minute. Step 11 (embeddings) needs TensorFlow and takes about half a minute.
- The notebook is self-contained: it loads the CSV and builds its own splits.

## Limitations

- The text is template-built (309-word vocabulary), so near-perfect accuracy reflects the data.
- The model cannot read words it never saw, including other languages.
- The stress test has 16 hand-written inputs, and the 0.40 confidence threshold is tentative.

## Related notebooks

The same text model is rebuilt in Part 5 and in the final notebook.
