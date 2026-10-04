# Part 1: Dataset Exploration and Preparation

## Overview

This section explores and prepares the datasets used for the EcoSort waste management system. The analysis covers three main data sources:

1. **RealWaste image dataset** – used for waste image classification.
2. **Waste descriptions dataset** – used for text-based waste classification.
3. **Waste policy documents** – used as the knowledge source for the later Retrieval-Augmented Generation (RAG) component.

The main goals were to understand the datasets, identify data quality issues, check for class imbalance and leakage, and create reliable training, validation, and test splits.

---

## 1. RealWaste Image Dataset

The RealWaste dataset contains images organized into nine waste categories.

### Waste Categories

The nine categories are:

- Cardboard
- Food Organics
- Glass
- Metal
- Miscellaneous Trash
- Paper
- Plastic
- Textile Trash
- Vegetation

A total of **4,752 images** were identified.

### Images per Class

The image classes are not perfectly balanced.

- **Plastic:** 921 images
- **Metal:** 790 images
- **Textile Trash:** 318 images

The largest-to-smallest class ratio is approximately **2.90**.

This imbalance is important because a model could perform better on the larger classes simply because it has more examples to learn from. Therefore, the image classification pipeline uses a stratified approach and considers class distribution during evaluation.

---

## 2. Image Size and Colour Mode

A sample of **40 images from each class**, giving 360 images in total, was examined.

All sampled images were:

- **524 × 524 pixels**
- **RGB**
- Square-shaped

Because the images are already square, resizing them to **224 × 224 pixels** does not introduce distortion.

---

## 3. Visual Inspection

Four random images from each waste category were examined to understand the visual characteristics of the dataset.

The images are generally top-down photographs taken on a light grey surface. The background does not appear to directly reveal the waste category.

Some differences were observed between categories:

- Food Organics and Vegetation often have a speckled appearance.
- Paper and Cardboard can look visually similar.
- Plastic and Miscellaneous Trash can overlap visually.
- Food Organics and Vegetation may also be difficult to distinguish.
- Miscellaneous Trash has less consistent visual characteristics.

Lighting also varies between images, including some images with a bluish appearance.

---

## 4. Image Quality

Brightness and sharpness were examined using a sample of 40 images per class.

Average brightness was relatively similar across the classes, ranging from approximately **150 to 162**.

Vegetation and Food Organics had higher sharpness scores, while Glass had the lowest average sharpness.

The sharpness measure also responds to texture. Therefore, the higher scores for Vegetation and Food Organics may partly reflect their naturally textured appearance rather than simply better image focus.

This is important because a neural network could potentially learn texture as a shortcut for identifying certain classes.

---

## 5. Duplicate Image Check

MD5 file fingerprints were used to identify exact duplicate image files.

Results:

- **4,752 images checked**
- **0 exact duplicate files**

This indicates that there were no byte-identical images in the dataset.

However, near-duplicate images, such as the same object photographed more than once, would not necessarily be detected by this method.

---

# 6. Waste Descriptions Dataset

The `waste_descriptions.csv` dataset contains **5,000 rows** and five text-related columns.

The main fields include:

- `description` – the text input
- `category` – the target label
- `material_composition`
- `common_confusion`
- `disposal_instruction`

The `description` column is used as the input for text classification, while `category` is the target.

---

## 7. Missing Values

The main missing data issue was found in `common_confusion`.

Approximately **50.08%** of this column is missing.

The missing values were distributed relatively evenly across categories, with proportions ranging from approximately **0.475 to 0.536**.

Each category has one distinct `common_confusion` note, suggesting that the missing values represent gaps in category-level information rather than meaningful absences.

---

## 8. Duplicate and Conflicting Descriptions

The dataset contained:

- **7 duplicate rows**
- **61 duplicate descriptions**
- **0 descriptions with conflicting category labels**

Duplicate descriptions were removed before creating the final text-data splits.

After removing duplicate descriptions:

- Original rows: **5,000**
- Unique descriptions: **4,939**

The class distribution remained almost unchanged after removing duplicates.

---

## 9. Class Balance

The text dataset is relatively well balanced.

- **Vegetation:** 600 rows
- **Paper:** 506 rows

The largest-to-smallest class ratio is approximately **1.19**.

This means that the text classification task does not have the same level of class imbalance as the image dataset.

A model that always predicted the largest class, Vegetation, would achieve only about **12% accuracy**, so meaningful classification requires learning information from the descriptions.

---

## 10. Label Leakage

The columns `material_composition`, `disposal_instruction`, and `common_confusion` contain information that can directly reveal the waste category.

These fields were therefore **not used as model inputs**.

Only the `description` field is used for text classification.

The other fields are retained for use in the later RAG/policy component.

---

## 11. Description Length and Vocabulary

The descriptions range from **1 to 10 words**, with most descriptions containing approximately 4–5 words.

There are:

- **21 one-word descriptions**
- **13 ten-word descriptions**

Short descriptions provide very little information for classification.

The dataset has a relatively small vocabulary of **309 words**.

The descriptions are largely template-based and commonly contain words describing:

- Items
- Materials
- Conditions
- Sizes
- Colours

Item-specific words can be useful for classification, but some material words can be misleading.

For example, the word "bottle" appears in descriptions belonging to both Glass and Plastic.

---

## 12. Misleading Material Words

Material keywords do not always correspond directly to the target category.

The proportion of occurrences where a material word appeared in another category was:

| Material word | Occurrences in another class |
|---|---:|
| Cardboard | 25.4% |
| Plastic | 23.9% |
| Paper | 21.8% |
| Metal | 20.7% |
| Glass | 10.1% |

This demonstrates why simply searching for material keywords would not be sufficient for reliable text classification.

---

# 13. Waste Policy Documents

The policy dataset contains **14 documents**.

All documents relate to the jurisdiction **Metro City**.

The documents range from approximately **86 to 146 words**.

Documents 1–9 generally focus on individual waste categories, while documents 10–14 cover multiple categories.

---

## 14. Policy Structure

Documents 1–8 generally contain sections such as:

- Acceptable Items
- Non-Acceptable Items
- Collection Method
- Preparation Instructions
- Benefits

The Miscellaneous Trash document has a different structure.

The multi-category documents begin with general requirements and then provide guidance for individual categories.

---

## 15. Policy Coverage

Policy coverage differs by waste category.

- Paper appears in 1 document.
- Food Organics appears in 2 documents.
- Glass, Textile Trash, Plastic and Cardboard appear in 3 documents.
- Vegetation, Metal and Miscellaneous Trash appear in 4 documents.

Paper therefore has the least policy coverage and no second document available as a fallback source.

---

## 16. Policy Consistency

The policy documents were checked for potentially conflicting information.

One example involved **window glass or mirrors**.

The item appears in:

- Document 2 under non-acceptable items.
- Documents 11 and 13 under Glass guidelines.

This creates a potential inconsistency between policy documents.

The category-specific document agrees with the corresponding note in the waste description dataset.

---

## 17. Policy and Description Vocabulary

The policy documents contain different terminology from the waste descriptions.

Only **69 of the 309 description vocabulary words** appear in the policy text.

Only about **28% of word occurrences** in the descriptions are represented in the policy vocabulary.

This suggests that directly matching a raw waste description against policy documents would not work reliably.

The planned approach is therefore to:

1. Classify the waste item first.
2. Identify its waste category.
3. Retrieve the relevant policy information using the predicted category.

---

# 18. Data Splitting

### Image Dataset

The original image dataset contains:

- **3,802 training images**
- **475 validation images**
- **475 test images**

The validation and test sets were created using a stratified split to preserve the class distribution.

A check confirmed:

- **0 images shared between validation and test**

This prevents the same image from appearing in both evaluation sets.

### Text Dataset

After duplicate removal, the text dataset contains **4,939 unique descriptions**.

The data was divided into:

- **80% training**
- **10% validation**
- **10% testing**

A stratified split was used to preserve the distribution of the nine waste categories.

Duplicate descriptions were removed before splitting so that the same description could not appear in multiple datasets.

---

# Conclusion

Part 1 established the quality and structure of the data used by the EcoSort system.

The analysis found that:

- The RealWaste image dataset contains 4,752 images across nine categories.
- Image classes are moderately imbalanced.
- The images are consistently 524 × 524 RGB in the sampled data.
- No exact duplicate image files were found.
- The text dataset contains 4,939 unique descriptions after duplicate removal.
- The text classes are relatively balanced.
- Several columns contain potential label leakage and were excluded from classification inputs.
- The policy documents contain some repeated and potentially conflicting information.
- Direct text-to-policy matching is unlikely to be reliable.
- Clean, stratified training, validation, and test splits were created for both image and text data.

These steps provide the foundation for the machine-learning and RAG components developed in the subsequent parts of the project.
