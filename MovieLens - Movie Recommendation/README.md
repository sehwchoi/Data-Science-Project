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
