# Leo Breiman (2001) — Random Forests: study notes

Paper: Leo Breiman, *Random Forests*, Machine Learning, 45, 5–32 (2001). DOI: 10.1023/A:1010933404324.

## Main idea in plain language

A Random Forest combines a large collection of decision trees. Each tree is grown with randomness: commonly, a bootstrap sample of training rows is used, and a random subset of input features is considered at each split. For classification, trees vote and the most-voted class is returned.

## Ingredient 1 — Strength of individual trees

A tree is strong when it tends to classify examples correctly and gives the correct class more votes than incorrect alternatives. If every tree is weak and usually wrong, majority voting cannot magically create reliable predictions. The forest needs useful individual learners.

Intuition: imagine asking 20 people who usually make informed judgments. Their individual opinions contain useful signal, so a majority vote can be dependable.

## Ingredient 2 — Low correlation between trees

Trees should not all make exactly the same mistakes. If every tree is nearly identical, the forest is just repeating one opinion many times. When tree errors are less correlated, voting can average out some individual errors and predictions become more stable.

Intuition: asking 20 people who all copied the same person's answer is not much better than asking one person. Asking independent people with useful knowledge gives a more reliable group vote.

## Why randomness is important

- Bootstrap sampling creates different training samples for different trees.
- Random feature selection at each split makes trees explore different predictors.
- These mechanisms encourage diversity and reduce the correlation between trees.

Randomness alone is not enough. Randomly generated poor trees do not help; the trees still need strength.

## What happens as more trees are added?

Breiman explains that the generalization error converges toward a limit as the number of trees becomes large. Therefore, more trees generally improve the stability of the forest estimate until the curve levels off, but they do not guarantee that each additional tree will improve test accuracy. Additional trees increase training and prediction cost.

## Link to the experiments

- The `n_estimators` graph shows how the test score changes as more trees are combined.
- Compare a single Decision Tree with a Random Forest to see whether voting reduces instability or improves test performance on your split.
- Compare with SVM to see that a non-tree decision boundary can behave differently.
- Do not claim a model is better only because it has a higher score on one small split; record the same fixed split and discuss limitations.

## Useful paper statements to remember

Breiman's abstract says forest generalization error depends on individual-tree strength and correlation between trees. In Section 2, he develops a bound involving average correlation and classifier strength. The paper also discusses bootstrap/out-of-bag estimates and random feature selection.

## Paper link

https://link.springer.com/article/10.1023/A:1010933404324

Free author PDF: https://stat-www.berkeley.edu/~breiman/randomforest2001.pdf
