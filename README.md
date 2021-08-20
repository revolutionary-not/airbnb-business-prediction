# Emotion detection from Audio

## Problem Statement
Understanding a customer's emotions during a service call is the key towards providing a great customer experience. This requires years of practice and a keen ear. Moreover, not all customers are explicit in conveying their true emotions. But when the volume of calls increase exponentially for a company, depending on the service agent to ascertain the customer's state of mind is not a scalable solution. There needs to be a more standardised, automated experience.

The intent of this project is to correctly identify the emotion of a customer during a call, so that the agent can speak in an appropriate manner and provide a solution. The model used in this notebook can be part of any call center product that agents use - they can use it to accurately pinpoint the emotion and provide an appropriate response, in order to increase the NPS scores.

## Data
The model has been trained on 2 datasets :

Surrey Audio-Visual Expressed Emotion (SAVEE) - https://www.kaggle.com/ejlok1/surrey-audiovisual-expressed-emotion-savee

Toronto Emotional Speech Set (TESS) - https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess

Both have been compiled with voice actors repeating certain words and phrases in 7 emotions - anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral.

## Approach
1. Used Mel Frequency Cepstral Coefficients(MFCC) as the feature since it is better suited for deep learning
2. Both datasets were trained separately on an Artificial Neural Network(ANN) and a Random Forest model for comparison
3. Since the data points were low for deep learning, both datasets were combined
4. Did dimensionality reduction using PCA, t-SNE and UMAP to bring down the number of features
5. Ran the new dataset with reduced features with 3 clustering algorithms - KMeans, Heirarchical & Gaussian Mixture Models
6. Trained the combined dataset on an ANN and compared the performance on other classifiers : Logistic Regression, Decision Tree, Random Forest, KNN & Gradient Boost

## Results
The combined dataset with MFCCs as the features returned a test accuracy of 96.3% for the ANN model

All other classifiers had a lower accuracy score

Out of the unsupervised clustering models, GMM had the best adjusted Rand score(0.31) and heirarchical clustering had the best silhouette score(0.31)
