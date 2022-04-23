# world-happiness-eda-ml repository

## About

This is a Mini-Project for SC1015 (Introduction to Data Science and Artificial Intelligence) which focuses on the happiness scores of countries from [World Happiness Report](https://www.kaggle.com/datasets/unsdsn/world-happiness) together with UN country statistics from [UNData 2015](https://data.un.org/) and [UNData 2017](https://www.kaggle.com/datasets/sudalairajkumar/undata-country-profiles). For detailed walkthrough, please view the source code in order from:

1. [Preliminary Test](https://github.com/keithsim-16/world-happiness-eda-ml/blob/main/1.%20Preliminary%20Test/Preliminary_Test.ipynb)
2. [Data Processing and Cleaning](https://github.com/keithsim-16/world-happiness-eda-ml/blob/main/2.%20Data%20Processing%20%26%20Cleaning/Data_Processing___Cleaning.ipynb)
3. [Exploratory Data Analysis](https://github.com/keithsim-16/world-happiness-eda-ml/blob/main/3.%20Exploratory%20Data%20Analysis/Exploratory%20Data%20Analysis.ipynb)
4. [Machine Learning](https://github.com/keithsim-16/world-happiness-eda-ml/blob/main/4.%20Machine%20Learning/Machine%20Learning.ipynb)

Things to Note:
- `2. Data processing and cleaning` contains a segment retrieving longitude and latitude data of countries and may take awhile to complete.
  
## Contributors

- @keithsim-16 - Sim Shi Jie, Keith (U2121044B)
- @priscillacs - Priscilla Celine Setiawan (U2123732G)
- @SongHeng3 - Teng Song Heng (U2122030K)

## Problem Definition

- Are we able to predict the happiness score of a country based on its attributes?
- Which model would be the best to predict it?

## Libraries Used

1. Numpy
2. Pandas
3. Seaborn
4. Matplotlib
5. Sklearn
6. Geopy  
7. Geopandas
8. Pydot
9. Scipy
10. Graphviz

If you experience any problems, please type or paste this code in the cmd or terminal to install the external libraries:
```
pip install geopy
pip install graphviz
```
or
```
pip3 install geopy
pip3 install graphviz
```
and
```
conda install -c conda-forge geopandas

If you experience any more problems:
https://geopandas.org/en/stable/getting_started/install.html
```

## Folder Summary
1. Preliminary Testing   
This folder contains a notebook with resources used to test the merging of 2 datasets with country names as the merging point. It is also used to primarily find out the most important predictors of happiness score as a reference to collect the same variables online but of a different year to be merged with the happiness score to increase the sample size of our dataset.

2. Data Processing & Cleaning  
This folder contains a notebook that merges data from 2015 and 2017 and performs data processing and cleaning.
  - It filters out 2015 UNData that is separated into 9 different csv files and combines them into a single dataframe.
  - 2015/17 UNData dataframe is combined with 2015/17 Happiness Report respectively.
  - 2015 and 2017 Data are combined together to form the final dataset.
  - Data is processed and cleaned and missing data is filled with KNN Imputer.

3. Exploratory Data Analysis  
This folder contains a notebook with the finalized dataset. The notebook explores on the various interesting insights that were discovered.

4. Machine Learning  
This folder contains a notebook that performs 3 machine learning models on the finalized data set. The notebook states the advantages and disadvantages of the models when using them with the current dataset, ultimately giving a conclusion on the best model to use for the current situation.

## Models Used

1. K Means Clustering
2. ElasticNet Linear Regression
3. Random Forests Regression

## Conclusion

- Regions experiencing the lowest happiness scores are in Africa, Middle East and South Asia.
- Scandinavian countries experience the highest happiness scores compared to the rest of the world.
- Employment type affects the internet usage of citizens in a country.
- As females in countries get more educated, the fertility rate decreases.
- Internet usage, life expectancy, employment in the services industry and secondary school enrolment ratio are the best in determining a country's happiness score. This was surprising as the group expected GDP per capita to be the most correlated at the beginning of the project.
- Ultimately we felt that random forests regression is the best machine learning technique for this project with 80.8% accuracy compared the K Means Clustering and ElasticNet linear regression.

## What did we learn from this project?

- Filling in missing data with k-nearest-neighbours imputation technique
- Machine learning techniques - K Means Clustering, ElasticNet and Random Forests regression from scikit-learn
- Importance of normalizing data before machine learning
- Collaborating using GitHub

## References

- <https://medium.com/@kyawsawhtoon/a-guide-to-knn-imputation-95e2dc496e>
- <https://scikit-learn.org/stable/user_guide.html>
- <https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c>
- <https://www.youtube.com/watch?v=Q81RR3yKn30>
- <https://www.youtube.com/watch?v=NGf0voTMlcs>
- <https://www.youtube.com/watch?v=1dKRdX9bfIo>
- <https://towardsdatascience.com/random-forest-in-python-24d0893d51c0>
- <https://neptune.ai/blog/random-forest-regression-when-does-it-fail-and-why>
- <https://developers.google.com/machine-learning/clustering/algorithm/advantages-disadvantages>
- <https://www.youtube.com/watch?v=EItlUEPCIzM&t=1229s&ab_channel=codebasics>
- <https://machinelearningmastery.com/elastic-net-regression-in-python/>
- <https://holypython.com/k-means/k-means-pros-cons/>
- <https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/>
- <https://www.datacamp.com/community/tutorials/k-means-clustering-python>
