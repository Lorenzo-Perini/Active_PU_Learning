# Active_PU_Learning

`Active_PU_Learning` (Active Positive and Unlabeled Learning) is a GitHub repository containing the **CAPe** [1] algorithm.

It refers to the paper titled *Class prior estimation in active positive andunlabeled learning*. 

Read the pdf here: [[pdf](https://www.ijcai.org/Proceedings/2020/403)].

## Abstract
Estimating the proportion of positive examples (i.e., the class prior) from positive and unlabeled (PU) data is an important task that facilitates learning a classifier from such data.  We explore how to tackle this problem when the observed labels were acquired via active learning. This introduces the challenge that the observed labels were not selected completely at random, which is the primary assumption underpinning existing approaches to estimating the class prior from PU data. We analyze this new setting and design **CAPe** [1], an algorithm that is able to estimate the class prior for a given active learning strategy. Our approach shows up being accurate and stable in recovering the true class prior.

## Contents and usage

The repository contains:
- CAPe.pyx, a function that allows to get the class prior estimate;
- Notebook.ipynb, a notebook showing how to use CAPe on an artificial dataset;
- evaluate_CAPe.py, a function used to get the experimental results on benchmark datasets;
- Experiments.ipynb, a notebook showing how to run the experiments.

To use CAPe, import the github repository or simply download the files. You can also find the benchmark datasets inside the folder Benchmark_Datasets or at this [[link](https://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/)].

Note that CAPe [1] has been developed in *Cython* (see Dependencies).

For examples of how to use the algorithm on benchmark datasets see the notebook inside `Experiments/`.

For further examples about the usage of the model *SSDO*, visit the following GitHub page [[anomatools](https://github.com/Vincent-Vercruyssen/anomatools)].

## Class prior estimation in Active Positive and unlabeled lEarning (CAPe)

Given a dataset with attributes **X**, the following procedure is followed.
First, the dataset is split into training and test sets using a stratified 5-fold crossvalidation. **All training data are initially unlabeled.**
Then iteratively, the user is queried until *k = 5* new labels are added to the training set in accordance with the **active learning strategy** and the probability of labeling the example correctly is equal to the *user's uncertainty*.
After adding  new labels to the training data, the class prior is estimated on the training data: 
1) **CAPe** creates the **global ranking**, where the higher is position, the higher is probability to be queried, 
2) it estimates the **propensity scores** using the counting techniques as explained in [1] and, 
3) the **class prior** is computed from the propensity scores.

Then, **SSDO** [2] classifier is retrained, and its performance on the test set is measured (using the estimated class prior to obtain binary predictions for the test data).
The process **stops** when 150 examples are labeled.

Given a (training) dataset **X_train** with partial labels *y* (where 1 means positive and 0 means unlabeled), and the number of labels **k** to be acquired in each iteration, the algorithm **CAPe** is applied as follows:

```python
from anomatools.models import SSDO
from CAPe import *

# Inizialize two lists for AL strategy:
query_list = []                             #List of queried examples
labeled_ex = np.zeros(n, dtype=np.int)      #List of labeled examples

# Inizialize all the parameters of our model
case = 0                                    #0 -> perfect oracle, 2 -> imperfect oracle
k = 5                                       #Number of labeled examples acquired at each step via AL strategy
prior_bet = 0.9                             #First bet of the prior
real_anomalies = np.where(y == 1)[0]        #If case = 2, you can use real_anomalies = []

# Estimate the density of examples
ker = KernelDensity().fit(X)
dmu = [np.exp(ker.score(X_train[i:i+1])) for i in range(n)]
mean_prob_term = math.log(np.mean(dmu),10)  #Take the log density

# Estimate the class prior with CAPe
prior, labeled_ex, query_list = CAPe(X_train, labeled_ex, query_list, k, real_anomalies,
				      1-prior_bet, mean_prob_term, case)
```

## Dependencies

The `CAPe` function requires the following python packages to be used:
- [Python 3](http://www.python.org)
- [Numpy](http://www.numpy.org)
- [Scipy](http://www.scipy.org)
- [Pandas](https://pandas.pydata.org/)
- [Cython](https://cython.org/)

## Contact

Contact the author of the paper: [lorenzo.perini@kuleuven.be](mailto:lorenzo.perini@kuleuven.be).


## References

[1] Perini, L., Vercruyssen, V., Davis, J.: *Class prior estimation in active positive and unlabeled learning.* In: Proceedings of the 29th International Joint Conferenceon Artificial Intelligence and the 17th Pacific Rim International Conference on Artificial Intelligence (IJCAI-PRICAI) (2020).

[2] Vercruyssen, V., Wannes, M., Gust, V., Koen, M., Ruben, B., Jesse, D.: *Semi-supervised anomaly detection with an application to water analytics.* In: Proceedings of 18th IEEE International Conference on Data Mining. pp. 527–536. IEEE (2018).


