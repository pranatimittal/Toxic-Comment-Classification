# Toxic-Comment-Classification
#### Problem Statement

Everyday while surfing the social media we encounter a lot of comments, reviews, tweets etc. that we believe might hurt the sentiments of the people of a particular group or a community. 
These comments are believed to be toxic in nature and can be categoried based on toxicity as - Toxic, Severe-toxic, Obscene, Threat, Insult, Identity_hate.
This is a Multi Label Classification problem which means that a given comment may belong to more than one category at the same time.
The aim of this project is to explore different aspects of Multi-Label classification and implementing those techniques to classify the comments on social media into various categories of toxicity. 
#### Bird's eye view of project

1. Getting Data: The dataset used for toxic comment classification is taken from Kaggle competition. Dataset has a large number of comments from Wikipedia talk page edits. They have been labeled by human raters for toxic behavior.
2. Exploratory data analysis (EDA):  EDA shows that the number of comments which were actually tagged were relatively low compared to the total number of comments in the dataset.
3. Data Pre-processing: First converted the comments to lower-case, used custom made functions to remove punctuation and non-alphabetic characters from the comments and transformed words to standered form by stemming. Then used TF-IDF to remove stopwords and convert comments to numerical vectors, after spliting dataset into train and test. 
4. Appling Multi-label Classification techniques:
   - Problem Transformation Methods:
     Single-Class Classifiers used are Multinomial Naive Bayes, Support Vector Machine and Random Forest Classifier. Parameters of these classifiers are tunned with help of Grid Search. Multi-label Clasification Algorithm used are:
     - Binary Relevance
     - Classifier Chains
     - Label Powerset
   - Adapted Algorithm:
     ML-kNN  which is derived from the traditional K-nearest neighbor (KNN) algorithm is used whose parameters are again tunned with Grid Search.
5. Comparing results and choosing best among them

#### Conclusion
After considering and comparing the various multi-label classification techniques on basis of accuracy score, we conclude that ML-kNN approach with k(number of neighbors)=7 gave best result of 90.8% accuracy score.
