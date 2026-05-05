# Rice Grain Dataset Project

This folder contains the rice dataset and evaluation scripts for the group project.

## Dataset
- `rice.csv` is the rice grain dataset.
- It has 7 numerical features and a binary label: `Cammeo` or `Osmancik`.

## Algorithms implemented
- `knn_rice.py` — k-Nearest Neighbors (scratch implementation)
- `decision_tree_rice.py` — Decision Tree (scratch implementation using Gini impurity)
- `random_forest_rice.py` — Random Forest (scratch implementation using entropy and bootstrap aggregation)

## Evaluation strategy
- Stratified 10-fold cross-validation
- Metrics: accuracy, precision, recall, and weighted F1-score
- Hyperparameter tuning with at least 6 settings per algorithm
- Hyperparameter performance tables and graphs
- For Random Forest: plot accuracy, precision, recall, and F1 versus number of trees

## Why these models?
- `k-NN` is a simple distance-based baseline and is easy to compare.
- `Decision Trees` are interpretable and handle numerical features with simple splitting.
- `Random Forests` often improve stability and generalization over single trees.

## Running the scripts
Use the project virtual environment from the repository root:

```bash
cd /Users/ashwinsayenju/Documents/ml_final
source .venv/bin/activate
python rice_project/knn_rice.py
python rice_project/decision_tree_rice.py
python rice_project/random_forest_rice.py
```

## Output files
Each script saves CSV evaluation tables and PNG graphs in `rice_project/`.
