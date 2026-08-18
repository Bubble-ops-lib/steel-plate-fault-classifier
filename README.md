# Steel Plate Fault Classification

**BITS Pilani WILP — M.Tech AI & ML — Machine Learning Assignment 2**
Author: Asmita Bera (2025AC05622)

---

## 1. Problem Statement

Surface defects on cold-rolled steel plates are traditionally identified by manual
visual inspection, which is slow, subjective and inconsistent between inspectors.
This project frames defect identification as a **supervised multi-class
classification** problem: given 27 geometric and luminosity measurements extracted
automatically from a scanned plate region, predict which of seven fault types is
present.

The practical value is throughput. An automated classifier lets a line flag and
route defective plates without stopping for human inspection, and lets rare but
costly defect types be tracked reliably rather than being lost in a general
"reject" bucket.

---

## 2. Dataset Description

**Source:** UCI Machine Learning Repository — Steel Plates Faults (Dataset ID 198)
https://archive.ics.uci.edu/dataset/198/steel+plates+faults

| Property | Value |
|---|---|
| Instances | 1,941 |
| Features | 27 (all numeric) |
| Target | `Fault_Type` — 7 classes |
| Missing values | None |
| Task | Multi-class classification |

Both assignment constraints are satisfied: 27 features (>= 12 required) and
1,941 instances (>= 500 required).

### Features

The 27 predictors fall into four groups:

- **Bounding geometry** — `X_Minimum`, `X_Maximum`, `Y_Minimum`, `Y_Maximum`,
  `Pixels_Areas`, `X_Perimeter`, `Y_Perimeter`
- **Luminosity** — `Sum_of_Luminosity`, `Minimum_of_Luminosity`,
  `Maximum_of_Luminosity`, `Luminosity_Index`
- **Plate context** — `Length_of_Conveyer`, `TypeOfSteel_A300`,
  `TypeOfSteel_A400`, `Steel_Plate_Thickness`
- **Derived shape indices** — `Edges_Index`, `Empty_Index`, `Square_Index`,
  `Outside_X_Index`, `Edges_X_Index`, `Edges_Y_Index`, `Outside_Global_Index`,
  `LogOfAreas`, `Log_X_Index`, `Log_Y_Index`, `Orientation_Index`,
  `SigmoidOfAreas`

Feature ranges span roughly seven orders of magnitude (`Y_Maximum` reaches
1.3 x 10^7 while several indices are bounded in [0, 1]), and two features
(`Orientation_Index`, `Luminosity_Index`) take negative values. Both facts
directly influenced preprocessing and model choice — see Section 4.

### Target distribution

In the raw dataset the target is encoded as seven mutually exclusive binary
indicator columns. These were verified to be mutually exclusive (every row sums
to exactly 1 across the seven flags) and then collapsed into a single
categorical `Fault_Type` column.

| Fault type | Count | Share |
|---|---|---|
| Other_Faults | 673 | 34.7% |
| Bumps | 402 | 20.7% |
| K_Scratch | 391 | 20.1% |
| Z_Scratch | 190 | 9.8% |
| Pastry | 158 | 8.1% |
| Stains | 72 | 3.7% |
| Dirtiness | 55 | 2.8% |

The dataset is markedly imbalanced — the largest class is over 12x the smallest.
`Other_Faults` is also a residual "none of the above" category rather than a
defect type with a coherent physical signature.

---

## 3. Links

- **GitHub repository:** https://github.com/Bubble-ops-lib/steel-plate-fault-classifier
- **Deployed Streamlit app:** https://steel-plate-fault-classifier13.streamlit.app

---

## 4. Methodology

### Train/test split

Stratified 80/20 split with `random_state=42`, giving 1,552 training and 389 test
instances. Stratification was necessary rather than optional: with `Dirtiness` at
2.8%, an unstratified draw could leave too few rare-class samples for per-class
metrics to be meaningful. After splitting, class proportions match to within
0.1 percentage points in both partitions.

### Preprocessing

All preprocessing is wrapped inside scikit-learn `Pipeline` objects so that
scalers are fitted on training folds only. This prevents test-set statistics from
leaking into training, and means each saved model is a single self-contained
artefact that the Streamlit app can apply to raw uploaded data.

| Model | Scaling | Reason |
|---|---|---|
| Logistic Regression | StandardScaler | Gradient descent is poorly conditioned across a 10^7 feature range |
| k-Nearest Neighbours | StandardScaler | Euclidean distance would otherwise be dominated by `Y_Maximum` alone |
| Decision Tree | None | Threshold splits are invariant to monotonic rescaling |
| Random Forest | None | Same as above |
| Gaussian Naive Bayes | None | Per-feature mean/variance estimation is scale-invariant |

**Choice of Naive Bayes variant:** `MultinomialNB` assumes non-negative
count-like features and is undefined on negative input. Because
`Orientation_Index` and `Luminosity_Index` both take negative values,
**GaussianNB** was used.

### Evaluation

Six metrics are reported: Accuracy, AUC, Precision, Recall, F1 and MCC.

For the multi-class setting, AUC uses one-vs-rest (`multi_class="ovr"`) on
predicted probabilities. Precision, Recall and F1 require an averaging strategy;
**macro averaging is reported as the headline result**, with weighted averaging
included for comparison. Macro was chosen because it weights every fault type
equally regardless of frequency — appropriate here, since detecting a rare defect
matters as much operationally as detecting a common one, and weighted averaging
would allow a model to coast on the majority class. MCC requires no averaging
parameter; it is computed once from the full 7x7 confusion matrix.

---

## 5. Models and Results

All metrics below are computed on the held-out test set (389 instances) using
macro averaging.

### 5.1 Logistic Regression

| Metric | Value |
|---|---|
| Accuracy | 0.7275 |
| AUC | 0.9379 |
| Precision | 0.7613 |
| Recall | 0.7305 |
| F1 | 0.7418 |
| MCC | 0.6487 |

Train accuracy 0.734 vs test 0.728 — a gap of only +0.006.

### 5.2 Decision Tree

| Metric | Value |
|---|---|
| Accuracy | 0.7429 |
| AUC | 0.8534 |
| Precision | 0.7531 |
| Recall | 0.7559 |
| F1 | 0.7530 |
| MCC | 0.6688 |

Train accuracy 1.000 vs test 0.743 — a gap of +0.257.

### 5.3 k-Nearest Neighbours (k = 5)

| Metric | Value |
|---|---|
| Accuracy | 0.7275 |
| AUC | 0.9184 |
| Precision | 0.7458 |
| Recall | 0.7482 |
| F1 | 0.7416 |
| MCC | 0.6548 |

Train accuracy 0.816 vs test 0.728 — a gap of +0.088.

### 5.4 Gaussian Naive Bayes

| Metric | Value |
|---|---|
| Accuracy | 0.4524 |
| AUC | 0.8386 |
| Precision | 0.4315 |
| Recall | 0.4795 |
| F1 | 0.3830 |
| MCC | 0.3789 |

Train accuracy 0.467 vs test 0.452 — a gap of +0.015.

### 5.5 Random Forest (Ensemble, 200 trees)

| Metric | Value |
|---|---|
| Accuracy | 0.7995 |
| AUC | 0.9640 |
| Precision | 0.8452 |
| Recall | 0.7793 |
| F1 | 0.8081 |
| MCC | 0.7400 |

Train accuracy 1.000 vs test 0.799 — a gap of +0.201.

Per-class breakdown:

| Fault type | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Bumps | 0.730 | 0.667 | 0.697 | 81 |
| Dirtiness | 0.889 | 0.727 | 0.800 | 11 |
| K_Scratch | 0.973 | 0.936 | 0.954 | 78 |
| Other_Faults | 0.711 | 0.837 | 0.769 | 135 |
| Pastry | 0.720 | 0.562 | 0.632 | 32 |
| Stains | 0.923 | 0.857 | 0.889 | 14 |
| Z_Scratch | 0.971 | 0.868 | 0.917 | 38 |

---

## 6. Comparison of All Models

### Macro averaging (headline)

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7275 | 0.9379 | 0.7613 | 0.7305 | 0.7418 | 0.6487 |
| Decision Tree | 0.7429 | 0.8534 | 0.7531 | 0.7559 | 0.7530 | 0.6688 |
| k-Nearest Neighbours | 0.7275 | 0.9184 | 0.7458 | 0.7482 | 0.7416 | 0.6548 |
| Gaussian Naive Bayes | 0.4524 | 0.8386 | 0.4315 | 0.4795 | 0.3830 | 0.3789 |
| **Random Forest** | **0.7995** | **0.9640** | **0.8452** | **0.7793** | **0.8081** | **0.7400** |

### Weighted averaging (for comparison)

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7275 | 0.9066 | 0.7329 | 0.7275 | 0.7277 | 0.6487 |
| Decision Tree | 0.7429 | 0.8283 | 0.7455 | 0.7429 | 0.7429 | 0.6688 |
| k-Nearest Neighbours | 0.7275 | 0.8941 | 0.7318 | 0.7275 | 0.7245 | 0.6548 |
| Gaussian Naive Bayes | 0.4524 | 0.7955 | 0.5741 | 0.4524 | 0.4003 | 0.3789 |
| **Random Forest** | **0.7995** | **0.9438** | **0.8062** | **0.7995** | **0.7993** | **0.7400** |

**Best performing model: Random Forest**, which ranks first on all six metrics
under both averaging strategies.

---

## 7. Observations

### 7.1 Bias and variance across the five models

The train and test results show a clear difference in how the five models behave. Logistic Regression has a very small train-test gap of only 0.006, while Gaussian Naive Bayes has a gap of 0.015. Although these small gaps might initially look positive, their relatively low accuracy shows that both models are underfitting the data and have high bias. In the case of Logistic Regression, this is expected because the model can only create linear decision boundaries, while the fault classes in this dataset are not linearly separable.
The other models show a different pattern. k-Nearest Neighbours has a larger gap of 0.088, while Random Forest and Decision Tree have gaps of 0.201 and 0.257 respectively. The Decision Tree shows the strongest sign of overfitting, with a training accuracy of 1.000 but a much lower test accuracy of 0.743. This suggests that the tree has effectively memorised the 1,552 training samples instead of learning patterns that generalise well to unseen data.


### 7.2 Why the ensemble beats the single tree

Both the Decision Tree and Random Forest achieve a training accuracy of 1.000, so looking only at the training results would make them appear equally strong. However, their test results tell a different story. The Decision Tree achieves 0.743 test accuracy and an MCC of 0.6688, while the Random Forest reaches 0.7995 accuracy and an MCC of 0.7400.
The main advantage of the Random Forest comes from combining many different trees instead of relying on one. In this case, 200 trees are trained using different bootstrap samples of the data. Each tree can make its own errors, but these errors are not exactly the same. When the predictions from all the trees are combined, many of the individual errors cancel each other out. As a result, the Random Forest keeps the strong learning ability of the individual trees while reducing the overall variance. This explains why it performs noticeably better on the test data even though both models fit the training data perfectly.


### 7.3 Why Naive Bayes fails on this dataset

Gaussian Naive Bayes gives the weakest overall performance, with an accuracy of 0.4524. This is only about 10 percentage points higher than the 34.7% majority-class baseline. Interestingly, its AUC is much higher at 0.8386, which suggests that the model is still able to identify useful patterns when ranking the possible classes, even though its final class predictions are often incorrect.
One reason for this behaviour is that several features describe very similar aspects of the same defect. For example, X_Minimum, X_Maximum, X_Perimeter, Pixels_Areas and LogOfAreas are all related to the size or geometry of the defect region. Naive Bayes assumes that the input features are conditionally independent given the class, but that assumption does not hold well here. Because related features provide overlapping information, the model can effectively count the same evidence multiple times and become too confident in an incorrect prediction.
This difference between accuracy and AUC is also a good example of why looking at only one evaluation metric would not give the complete picture. The six reported metrics provide a much better understanding of how each model is actually behaving.


### 7.4 Where the remaining errors are concentrated

The Random Forest performs particularly well on some defect types. K_Scratch has an F1 score of 0.954 and Z_Scratch reaches 0.917, showing that these two types are being identified quite reliably. On the other hand, Pastry and Bumps are more difficult, with F1 scores of 0.632 and 0.697 respectively.
Other_Faults is an interesting case. It has a relatively high recall of 0.837, but its precision is lower at 0.711. The confusion matrix shows that predictions are moving in both directions between Other_Faults and the more diffuse texture-related defects. This suggests that Other_Faults is acting somewhat like an attractor for defects that do not have a very distinctive pattern. In other words, the class itself is not necessarily the hardest one to recognise, but its broad definition makes it easier for the less distinctive defect types to be confused with it.
Another interesting result is that the macro F1 score of 0.8081 is actually higher than the overall accuracy of 0.7995. This indicates that the model is performing reasonably well across the different classes, including the less frequent ones. This is different from the pattern often seen with imbalanced datasets, where a model can achieve good overall accuracy mainly by performing well on the majority class.


### 7.5 Which model to deploy, and why

Based on the results, Random Forest is the strongest choice for deployment. It ranks first across all six evaluation metrics under both macro and weighted averaging. Its test accuracy is 0.7995, with an AUC of 0.9640, F1 score of 0.8081 and MCC of 0.7400. This makes it the most reliable overall model among the five tested approaches.
There are still some practical trade-offs to consider. The Random Forest takes longer to train, at around 0.56 seconds compared with 0.02 seconds for the Decision Tree, and its saved model is also larger at about 11 MB. The results for Dirtiness and Stains should also be interpreted carefully because they are based on only 11 and 14 test samples respectively. With such small sample sizes, the measured performance for these classes has greater uncertainty.
Even with these limitations, Random Forest is the best option when prediction performance is the main priority. The Decision Tree could still make sense in a situation where simpler interpretation or a smaller model size is more important than achieving the highest possible accuracy.


---

## 8. Streamlit Application

Deployed at https://steel-plate-fault-classifier13.streamlit.app

Features:

1. **Dataset upload** — CSV uploader in the sidebar; falls back to the bundled
   `test_data.csv` when no file is supplied. Uploaded files are validated against
   the expected 27 feature columns before prediction.
2. **Model selection** — dropdown covering all five trained classifiers.
3. **Evaluation metrics** — all six metrics displayed for the selected model,
   with a macro/weighted averaging toggle.
4. **Confusion matrix and classification report** — 7x7 confusion matrix plus a
   full per-class precision/recall/F1 breakdown, followed by a table comparing
   all five models on the same uploaded data.

---

## 9. Repository Structure

    steel-plate-fault-classifier/
    ├── app.py                          Streamlit application
    ├── steel_plate_fault_models.ipynb  Full analysis: EDA, training, evaluation
    ├── steel_plates_faults.csv         Complete dataset (1941 x 28)
    ├── test_data.csv                   Held-out test set (389 x 28)
    ├── requirements.txt                Pinned dependencies
    └── model/
        ├── logistic_regression.joblib
        ├── decision_tree.joblib
        ├── knn.joblib
        ├── naive_bayes.joblib
        ├── random_forest.joblib
        ├── metadata.json               Feature names, class names, sklearn version
        ├── results_macro.csv
        └── results_weighted.csv

---

## 10. Reproducing

    pip install -r requirements.txt
    jupyter notebook steel_plate_fault_models.ipynb   # retrain and re-save models
    streamlit run app.py                              # launch the app locally

All randomness is seeded with `random_state=42`, so results are reproducible.
Models were trained with scikit-learn 1.7.2, and `requirements.txt` pins that
version so the saved pipelines unpickle correctly on Streamlit Cloud.
