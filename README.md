# Perfguard
An implementation of Perfguard.

---
## Data description

We leverage the feature generation method from the source code of  [Lero](https://github.com/AlibabaIncubator/Lero-on-PostgreSQL) to preprocess our plans as trainging data set and testing data set.


## How to run

1. Modify necessary configuration about training and testing in `ImportConfig.py`.
2. Train and test
```bash
# 1. train
python train.py
# 2. test
python test.py
```