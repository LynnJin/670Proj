# 670Proj

This project focuses on the numerical experiments in the paper ["Robust Solutions of Optimization Problems Affected by
Uncertain Probabilities"](https://doi.org/10.1287/mnsc.1120.1641).

## Experiment ideas
* Data Sampling Process\
I assume there is a true distribution. I sample N data from that distribution to calculate the empirical distribution and use it to build the uncertainty set. I sample the testing distribution around the true distribution at the confidence level of 0.05.
* Influences of Confidence level\
For the problem with small sample data size, I test how the out of sample performance and the reliability change with the confidence level \alpha. I treat the mean return of several sample distributions as the indicator of the out of sample performance. The ratio that the return with sample distribution and robust solution is lower than the robust problem's objective value represents the reliability.
* Cross Validation\
 I use 2-fold method to select the confidence level when the sample size is small(less than 50). Because the size of the data size is small, the 5-fold method doesn't work well. I try to find the confidence level, which will keep the reliability greater than 90% while maximizing the mean return.
## Code structure
I have five python files in the project.\
* [data.py](https://github.com/LynnJin/670Proj/blob/main/data.py)\
This file includes sampling data, sampling distribution and generates the candidate alpha for testing.
* [model.py](https://github.com/LynnJin/670Proj/blob/main/model.py)\
This file includes the functions for building the deterministic model and robust model.
* [evaluate.py](https://github.com/LynnJin/670Proj/blob/main/evaluate.py)\
This file includes solving the optimization problems and calculate the mean, range, and reliability using sample distributions.
* [main.py](https://github.com/LynnJin/670Proj/blob/main/main.py)\
This file includes the functions for a sanity check, out of sample performance testing, and cross validation.
* [figure.py](https://github.com/LynnJin/670Proj/blob/main/figure.py)\
This file is for drawing figures.
