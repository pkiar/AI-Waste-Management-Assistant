# EcoSort: Waste Management Assistant: (Part4: RAG Generator)
Overview

EcoSort is a waste management assistant developed as part of a series of machine learning and deep learning labs.

The project builds toward an integrated waste management system for identifying waste and providing appropriate recycling and disposal guidance.

Other Labs

Before the detailed RAG work, the project covered several related areas:

Part 1: Data exploration and preparation.
Part 2: Machine learning and classification.
Part 3: Neural networks and text-based modelling.
Part 5: Integration of the developed components into the EcoSort waste management system.

These parts provided the foundation for the main work presented in Part 4: Recycling Instruction Generation with RAG.

# Part 4: Recycling Instruction Generation with RAG

## System Workflow

The system follows this process:

```text
Waste Category / Question
          ↓
Policy Documents
          ↓
Policy Chunking
          ↓
Hybrid Retrieval
          ↓
Relevant Policy Context
          ↓
Fine-Tuned FLAN-T5
          ↓
Generated Recycling Instructions
          ↓
Section-Aware Checker
          ↓
Final Answer
```

## Dataset

The RAG system uses:

* `waste_policy_documents.json`
* `waste_descriptions.csv`

The policy dataset contains **14 Metro City policy documents** covering the nine waste categories.

The documents were divided into **54 smaller policy chunks** to make retrieval more targeted.

## Hybrid Retrieval

Two retrieval approaches were implemented and compared:

### TF-IDF

TF-IDF provides keyword-based matching and performed particularly well when the waste category was explicitly mentioned.

### MiniLM Embeddings

The `all-MiniLM-L6-v2` Sentence Transformer was used to create semantic embeddings with **384 dimensions**.

This allowed the system to identify policy text based on meaning rather than only exact words.

### Final Hybrid Model

The two retrieval scores were combined.

The final configuration used:

```text
TF-IDF: 25%
MiniLM: 75%
```

This configuration gave the best overall performance during the alpha evaluation.

### Retrieval Evaluation

The retrieval system was tested using **54 questions**:

* 36 category-based questions
* 18 item-based questions

The final hybrid retrieval achieved:

* **Category MRR: 1.000**
* **Category Hit@1: 1.000**
* **Item MRR: 0.944**
* **Item Hit@1: 0.889**
* **Hit@3: 1.000**
* **Hit@5: 1.000**

The retrieval completeness test showed that **k=6 retrieved all required policy chunks for all nine categories**, giving **100% completeness**.

# Fine-Tuning the Generation Model

The generation model used was:

**`google/flan-t5-small`**

The model has approximately **77 million parameters**.

Training examples combined a question with the required policy context and distractor chunks from other categories.

The main training experiment used:

* **216 training examples**
* **54 validation examples**
* **3 epochs**
* AdamW optimizer
* Gradient clipping

### Training Results

| Epoch | Training Loss | Validation Loss |
| ----- | ------------: | --------------: |
| 1     |        0.2904 |          0.0286 |
| 2     |        0.0326 |          0.0116 |
| 3     |        0.0157 |          0.0068 |

Both training and validation loss decreased during training.

The training run took approximately **33.1 minutes on CPU**.

The trained model and tokenizer were saved in the `waste_rag_model` folder.

# Context-Use Testing

The model was tested to determine whether it actually followed the policy information supplied in its context.

Two tests were performed:

### Test A: Changing Policy Information

A policy statement was changed and the model was asked to generate a new answer.

Across all nine categories:

* New information was used in **9/9 categories**.
* Old information was removed in **9/9 categories**.

### Test B: Removing Policy Information

A policy section was removed from the retrieved context.

The model produced **0 invented items** across all nine categories.

However, some answers contained repeated or incorrectly positioned information.

This showed that the model was using the supplied context, but additional checking was necessary.

# Section-Aware Checker

A section-aware checker was created to detect problems in generated answers.

The checker verifies that:

1. Headings are not repeated.
2. Policy items appear under the correct heading.
3. Information from sections that were not retrieved does not appear in the answer.

If the generated answer fails the checks, the system can fall back to a reference answer created from the policy information.

The final `generate_instructions(category)` function:

* Accepts a waste category.
* Retrieves up to six relevant policy chunks.
* Generates recycling instructions.
* Checks the generated answer.
* Provides a fallback when necessary.
* Handles unknown categories.

All nine valid categories returned:

```text
ok
```

An unknown category such as `Furniture` was correctly rejected.

An intentionally incorrect Paper answer triggered **7 checker problems**, showing that the checker could identify repeated headings and incorrectly placed policy information.

# Full RAG Pipeline Results

The complete pipeline was tested across all nine categories.

Every category achieved:

```text
Exact Match: 1.0
Token F1:    1.0
Extra words: []
Missing words: []
```

This showed that retrieval and generation worked correctly together for the tested reference-based examples.

# Generation Method Comparison

Six generation methods were tested:

* Greedy decoding
* Beam search
* Temperature sampling
* Top-p sampling
* Top-k sampling

For the tested examples, all methods produced the same results:

* **Exact Match: 1.0**
* **Token F1: 1.0**
* **Trigram support: 0.634**
* Approximately **101 words per answer**

Changing the generation settings did not change the output in this evaluation.

# Unseen Category Experiment

**Paper was held out as a training target category** to test whether the model could use retrieved policy information for a category that was not used as a training target.

The held-out experiment used:

* **192 training examples**
* **48 validation examples**
* **0 Paper reference answers as training targets**

When tested using the real Paper policy chunks:

```text
Exact Match: False
Token F1: 0.853
```

The answer was mostly relevant, but the checker detected **6 incorrectly placed items**.

This demonstrated that the model could make use of retrieved policy context for the held-out category, while also showing the importance of the checker.

# Readability Evaluation

The generated answers were evaluated for readability.

Average results were:

* **10.54 words per sentence**
* **39.86 Flesch reading ease**
* **10.31 estimated grade level**
* **0.11 near-duplicate sentence pairs**

The answers were reasonably structured, although the reading-level results suggest that some instructions may be relatively difficult for general users.

# Limitations

* Exact-match results are based on highly templated reference answers.
* The retrieval evaluation uses a relatively small hand-written question set.
* The unseen-category experiment only held out Paper.
* The checker cannot detect every possible generation error.
* Some generated answers contained repeated or misplaced information.
* Training was performed on relatively small datasets using CPU resources.

# Conclusion

This project demonstrates a complete **Retrieval-Augmented Generation pipeline for waste recycling instructions**.

The system combines **policy chunking, TF-IDF retrieval, MiniLM semantic embeddings, hybrid retrieval, FLAN-T5 fine-tuning, and a section-aware checker**.

The final pipeline achieved **100% exact match and 100% token F1 across the nine tested categories**. The context-use experiments also showed that the model responded to changes in the retrieved policy information.

At the same time, the unseen Paper experiment and checker results demonstrated that good retrieval and generation scores do not eliminate the need for validation.

Overall, the project shows how RAG can be used to generate **policy-grounded recycling instructions** while providing additional checks to improve reliability.

