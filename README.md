# EcoSort Waste Management Assistant

An assistant that takes a photo and/or a short description of a waste item and returns its category, recycling instructions, and the policy text those instructions came from. Built for the Module 8 summative lab.

## Team Members and Roles

This project was developed by a team of five members, with each member responsible for a specific part of the project:

| Member               | Role                                             | Responsibility                                                                                                                                                       |
| -------------------- | ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Francis Mwaura**   | Part 5 – Integrated Assistant & Project Compiler | Integrated all components of the project into the final assistant, compiled the complete project, and worked on the final end-to-end pipeline.                       |
| **Faith Njau**       | Part 1 – Data Exploration & Preparation          | Explored and prepared the datasets, identified data quality issues, handled data splitting, and prepared the data for modelling.                                     |
| **Peter Kiarie**     | Part 2 – CNN Classifier                          | Developed and evaluated the image classification model using CNN/MobileNetV2 transfer learning for waste image classification.                                       |
| **Fascos Jepleting** | Part 4 – RAG Generator                           | Developed the Retrieval-Augmented Generation (RAG) component, including retrieval, fine-tuning, grounding tests, and evaluation of generated recycling instructions. |
| **Kevin Mukundi**    | Part 3 – Simple Text Classifier                  | Developed the text classification component using TF-IDF and LinearSVC to classify waste descriptions into the nine waste categories.                                |

The project combines the work from all five parts into one integrated waste-management assistant.

## Project Architecture

It joins three models:

| Model | What it does | Built with |
|---|---|---|
| Image classifier | Photo to one of 9 waste categories | MobileNetV2 transfer learning (TensorFlow / Keras) |
| Text classifier | Description to one of 9 categories | TF-IDF with a calibrated LinearSVC (scikit-learn) |
| Instruction generator | Category to recycling instructions, grounded in policy documents | Hybrid retrieval plus a fine-tuned `flan-t5-small` (PyTorch / transformers) |

## Where to look

| File | Purpose |
|---|---|
| `ecosort_final.ipynb` | **The final pipeline.** Builds the final models and runs the assistant. Every choice has a pointer to its evidence. |
| `notebooks/part1_exploration_and_preparation.ipynb` ([README](notebooks/README_part1.md)) | Data exploration, the split problem and its fix, pipelines |
| `notebooks/part2_cnn_classifier.ipynb` ([README](notebooks/README_part2.md)) | The CNN: model, training, mistakes, tuning |
| `notebooks/part3_text_classifier_simple.ipynb` ([README](notebooks/README_part3.md)) | The text classifier: baselines, cross-validation, mistake analysis, embeddings |
| `notebooks/part4_rag_generator.ipynb` ([README](notebooks/README_part4.md)) | Retrieval, fine-tuning, grounding tests, sampling, checker, held-out category |
| `notebooks/part5_integrated_assistant.ipynb` ([README](notebooks/README_part5.md)) | Confidence study, the assistant, end-to-end test, feedback |

The part notebooks show **how each decision was reached**. The final notebook shows **what was built**.

## Setup

Python 3.13 (the versions below were used on Windows 11, CPU only).

```
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Put these next to the notebook (they are not in the repository):

- `RealWaste/` : the RealWaste image dataset, one sub-folder per class (Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, Paper, Plastic, Textile Trash, Vegetation).
- `waste_descriptions.csv` and `waste_policy_documents.json` : the course data files.

## Running `final_notebook.ipynb`

- **Run all**, or run the two halves separately. Part A (Steps 1 to 12) uses TensorFlow. Part B (Steps 13 to 27) uses PyTorch and TensorFlow and only needs files from , so you can restart the kernel between the halves to save memory.
- Everything the notebook creates goes into : `waste_cnn.keras`, `waste_rag_model/`, `text_model.joblib`, `text_splits.csv`, `image_test_paths.csv`.
- If `waste_rag_model` exists, the generator is loaded instead of fine-tuned again. Set `FORCE_RETRAIN = True` to train again.
- Approximate time on a CPU, from the part notebooks: Part A about 10 minutes; Part B about 25 minutes the first time (the generator fine-tuning is about 20), a few minutes afterwards.
- Images are read from file paths in batches and nothing large is cached, to keep memory use low.

## Main results (from the part notebooks)

| Part | Result |
|---|---|
| Data | The provided image split leaked 129 images between validation and test; fixed. Three CSV columns leak the label, so only `description` is used. |
| CNN | Test accuracy 0.8168. Weakest classes: Miscellaneous Trash (recall 0.57) and Plastic (0.76). One run varies by about 2 points. |
| Text classifier | Test accuracy 1.0 on template-built descriptions (a property of the data, not real-world accuracy). |
| Generator | Reproduces all nine references; follows an edited fact in 9 of 9 categories; invents no missing items; a model trained without Paper still wrote the exact Paper answer. A section-aware checker catches an inverted list that word-overlap metrics miss. |
| Assistant | CNN confidence cut-offs 0.80 and 0.70 chosen from measured accuracy by confidence (93% above 0.8, about 54% below 0.7). In 360 requests, every confident CNN error became a request to confirm. About 0.11 s per request. |

The final notebook prints its own numbers. They can differ slightly from the part notebooks (training is not perfectly repeatable, and the final notebook splits the images differently).

## Limitations

- The descriptions, reference answers and policy documents are template-built, so near-perfect text and generation scores show the logic, not real-world accuracy.
- The CNN cannot recognise pictures that are not waste (only synthetic examples were tested).
- When a photo is unsure, a description is trusted without a check. The feedback guard cannot detect a wrong correction, and the feedback log is not saved.
- Single runs on a CPU and small test sets.

## Data and licence notes

The RealWaste dataset has its own licence and terms; download it from its source. Check that you may publish the course CSV and JSON files before adding them to a public repository. The saved models are not committed (`artifacts/` is ignored); the notebook recreates them.

