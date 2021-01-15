# movie-recommendation-engine
A Context based Recommendation system for Big Data systems to recommend movies and TV shows for users.

# Movie recommendations for users

## TABLE OF CONTENTS

* [Objective](#objective)
* [Data](#data)
* [Technologies](#technologies)
* [Algorithms](#Algorithm)
* [Implementation](#implementation)
* [Results](#results)
* [References](#references)

## OBJECTIVE
The main objective of the project is to design a full fledge custom movie-recommendation engine for the users, the other key objectives are
1. Design a content-based recommendation system that provides movie recommendations to users based on movie genres
2. Implement a collaborative-filtering approach to recommend movies to users

## DATA
In view of achieving the core objectives using multiple approaches, two different data sources were referred.

1. [Movielens data](http://files.grouplens.org/datasets/movielens/ml-latest.zip): Consisting of 27 million instances of movie ratings provided by users

2. [Movies metadata](https://www.kaggle.com/rounakbanik/the-movies-dataset): Movie metadata with 24 features capturing various details about the film

## TECHNOLOGIES
Python - Spark, pyspark, sklearn, nltk, scikit learn, pandas, matplotlib, seaborn

## ALGORITHMS
- Collaborative Filtering using ALS algorithm
- Content based filtering using k-means clustering

## IMPLEMENTATION
### Collaborative filtering using ALS algorithm:
Collaborative filtering technique allows filtering out items that a user might like by leveraging
the ratings of similar users. The underlying assumption in recommendation using collaborative
filtering is that, if the user A and user B share a similar response (movie rating in our case) to a
movie, then they are likely to share a similar response to any movie X, compared to any random
user.

- Employed the model-based system of performing collaborative filtering on the MovieLens dataset. 
- Implemented Alternating Least Square(ALS) with Spark. ALS is a matrix factorization technique to perform collaborative filtering. The
objective function of ALS uses L1 regularization and optimizes the loss functions using Gradient Descent. 
- The dataset contained movie_id and user_ratings in the format of a user-rating matrix shown as factors as given below:

![Capture1](https://user-images.githubusercontent.com/9445072/104695525-c0e76880-56c1-11eb-9924-fb9d185d51c4.JPG)

Here, d would be the number of features we learned from each user and movie association. With ALS, we intend to minimize the error in the matrix calculation shown below:

![Capture1](https://user-images.githubusercontent.com/9445072/104695740-0ad04e80-56c2-11eb-8cf2-e6e72f6387b4.JPG)

And the error is given by the below equation:

![Capture1](https://user-images.githubusercontent.com/9445072/104695844-318e8500-56c2-11eb-9ee6-21f72a08d969.JPG)

We train the ALS model by tuning the below hyper-parameters:
- Rank: Indicating the number of latent factors generated in matrix factorization
- regParam: The L1-regularization parameter used in ALS algorithm
- maxIter: The maximum number of iterations the algorithm is run

After tuning the parameters and implementing ALS with Cross validation an optimal RMSE value of 0.8037 for 30 latent factors at the regParam value of 0.05 in 10 iterations.

Below are the resulting movie predictions made by the tuned ALS model on the test data 

![Capture1](https://user-images.githubusercontent.com/9445072/104697307-67cd0400-56c4-11eb-9b8a-b9ebfd0e71ef.JPG)

Refer to this link for code - [Collaborative filtering using ALS](https://github.com/abhilashhn1993/movie-recommendation-engine/blob/main/Code/ALS_model.ipynb)

### Context-based filtering using k-means clustering:
- Used the movies-metadata file with 45k instances and 24 features. In view of capturing the content-based information for a given movie, the feature 'Overview' which provides the description about the genre as well as the plot of the film
- The description containing a paragraph with average 50-70 words was cleaned to remove whitespaces and stopwords were removed
- The text data is then input to compute TF-IDF scores and the corresponding TF-IDF matrix is generated
- The scores are used to group similar movies (content with similar scores) into clusters
- These clusters provide recommendations to user

Below is a sample output of movie recommendations provided by the k-means clustering

![Capture1](https://user-images.githubusercontent.com/9445072/104699109-18d49e00-56c7-11eb-980d-b0b9a6f526d3.JPG)

Refer to this link for code: [Context-based filtering using k-means clustering](https://github.com/abhilashhn1993/movie-recommendation-engine/blob/main/Code/K_means_Clustering.ipynb)

## RESULTS
The movie recommendation system has shown tremendous potential. Movie recommendations have been pretty accurate for specific users, and movie titles have been successfully segmented into clusters based on their overview content. In the future scope, I plan to extend project to build recommender systems for TV shows

## REFERENCES
- https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-1-knn-item-based-collaborative-filtering-637969614ea
- https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-2-alternating-least-square-als-matrix-4a76c58714a1
- https://realpython.com/build-recommendation-engine-collaborative-filtering/