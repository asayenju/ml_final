# Extra Credit (EC3 + EC4)

This folder contains the EC3/EC4 10-fold runner and result artifacts.

## Implementations

### EC3: `HeterogeneousBootstrapEnsembleEC3`
- Location: `library/ensemble.py`
- Structure: 5 bootstrap-trained members
1. NN member (architecture 1)
2. NN member (architecture 2)
3. NN member (architecture 3)
4. Random forest member (RF #1)
5. Random forest member (RF #2)
- Combiner: unweighted majority vote over member predictions.

### EC4: `RandomForestClassifierErrorSplitScratch`
- Location: `library/gradient_boosting.py`
- Type: random forest classifier variant (not gradient boosting).
- Split criterion: **classification-error reduction**:
  - Parent error: `1 - max_class_probability(parent)`
  - Child objective: weighted child classification error
  - Gain: parent error minus weighted child error
- Explicitly not using entropy/information gain/gini for split scoring.

## Prompt-intent alignment check

### EC3 intent check (heterogeneous ensemble: 3 NNs + 2 RF + bootstrap + majority vote)
- Status: **Yes**.
- Why: implementation trains 3 NN members plus 2 RF members, each on bootstrap samples, then predicts by majority vote.

### EC4 intent check (`RandomForestClassifierErrorSplitScratch` with classification-error split reduction)
- Status: **Yes**.
- Why: the class computes split gain from classification-error reduction and uses it for numeric/categorical split selection.

## Caveats to note

1. `extra_credit/table_results_ec3_ec4_10fold.csv` currently labels EC4 rows as `ec4_stochastic_gradient_boosting`, which does not match the current runner/class naming (`ec4_rf_error_split` / `RandomForestClassifierErrorSplitScratch`). Treat those labels as stale/misnamed output metadata.
2. In the current runner, EC3 uses a shared one-hot encoder for NN members, while the RF member operates on raw object-typed features with internal numeric-column inference. This is intentional but means preprocessing differs by member family.
3. Majority vote in EC3 is unweighted and tie behavior is determined by `np.unique(...)+argmax` ordering, not by calibrated confidence scores.

## Explicit question: does gradient boosting combine multiple different algorithms in one?

**No, generally it does not.** Standard gradient boosting combines many weak learners of the same base family (commonly shallow trees) in sequence. A heterogeneous mix of different algorithms in one ensemble is more characteristic of stacking/voting-style designs like EC3.

## Run command

From project root:

```bash
python extra_credit/run_ec3_ec4_10fold.py
```

Primary output:
- `extra_credit/table_results_ec3_ec4_10fold.csv`
