# Perfguard

An implementation of [Perfguard](http://www.vldb.org/pvldb/vol14/p3362-hossain.pdf).

---
## Data description


We leverage the plan generation method from [Lero repo](https://github.com/AlibabaIncubator/Lero-on-PostgreSQL) to generate plans as trainging data set and testing data set. Moreover, the feature generation method of Lero are used to extract features from plans.


## How to run

1. Configure python runtime environment
```bash
pip install -r requirement.txt
```
2. Modify necessary configuration about training and testing in `ImportConfig.py`.
3. Train and test
```bash
# 1. train
python train.py
# 2. test
python test.py
```