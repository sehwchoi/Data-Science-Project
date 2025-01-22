# Movie Lens Movie Rating Prediction

## About the project

Predicting a movie rating that a user would give based on KNN/Truncated SVD models.

## Data

Number of rating instances  90000
Number of unique users:  943
Number of unique items:  1675

## Models
* Baseline
  * based on the average of the selected movie

* Collaborative Filtering
  * KNN Basic, KNN with Z score
    * mean squared differences, cosine similarity, pearson correlation for the similarity measurements. Pearson correlation based model worked best.
    * 4, 6, 8, 10, 12 are tested as the number of neighbors
    * item based or user based.
* Truncated Singular Value Decomposition
    * Predicting the rating by finding the weights on the characteristics of the movie and users and decomposing the user-item matrix into three matrices. The number of characteristics is called latent factors. By performing matrix factorization, the rating can be estimated for given a user and an item.
    * 20, 40, 60, 80 are tested as the number of latent factors. 80 worked based
    * Weights on the latent factors are calculated using stochastic gradient descent. sampling, loss function, derivative of the loss function for finding local minimum


## Discussion/Results:

* SVD(80 latent factors) - 0.90 RMSE
* KNN with Zscore(Pearson correlation,  12 k) - 0.95 RMSE
* KNN Basic with User based(Pearson correlation,  12 k) - 0.96 RMSE

The SVD model outperformed others, followed by KNN with Z-score and KNN with user-based approaches. For the SVD model, performance improved as the number of latent factors increased. In the case of KNN, various similarity metrics were tested, with Pearson correlation producing the lowest error rates.

In real-world scenarios, the user and item matrix can often be highly sparse, meaning that only a few users actively provide ratings. This poses a challenge for the SVD model, as the missing items makes it difficult to accurately estimate the strength of latent factors.

Additionally, the cold start problem, involving predicting ratings for new users, can be a hurdle. While in this dataset, all users in the dev data are seen in the train data, this might not be the case in reality.

To address these challenges, it would be more practical to build a hybrid model. This hybrid model could utilize SVD for users with a lot of historical ratings while predicting ratings for others based on either historical averages on movies or their content/meta information.
