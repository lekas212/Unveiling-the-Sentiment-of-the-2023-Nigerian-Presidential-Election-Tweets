#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis of the 2023 Nigerian Presidential Election Tweets# 

# # Data Preparation

# In[1]:


get_ipython().system('pip install wordcloud')
get_ipython().system('pip install textblob')



# In[2]:


import re
import string
import itertools
import numpy as np
import pandas as pd
import seaborn as sns




from tensorflow.keras.preprocessing import sequence, text
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from bs4 import BeautifulSoup
from tensorflow.keras.optimizers import Adam
 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer



import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from  nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import nltk
import warnings
warnings.simplefilter("ignore")
from nltk.corpus import stopwords
from sklearn.ensemble import StackingClassifier


  
  
DATASET_ENCODING = "ISO-8859-1"

POSITIVE = "Positive"
NEGATIVE = "Negative"
NEUTRAL = "Neutral"
SENT_THRESH = (0.4, 0.7)


# In[3]:


nltk.download('stopwords')
nltk.download('vader_lexicon')
stemmer = nltk.stem.PorterStemmer()
stop_words = stopwords.words("english")
FORMATTED = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
from nltk.tokenize import TweetTokenizer
nltk.download('punkt')
nltk.download('wordnet')

nltk.download(['omw-1.4'])



# # Loading the data

# In[4]:


df =pd.read_csv('nigerian_presidential_election_2023_tweets.csv', encoding =DATASET_ENCODING)


# # Data Exploration

# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df = df.iloc[0:5000, :]


# In[8]:


df.tail()


# In[9]:


df.dtypes


# In[10]:


df.info()


# In[11]:


df.describe()


# In[12]:


df.corr()


# In[13]:


df.isna().sum()


# In[14]:


df = df.fillna('Unknown')


# In[15]:


df.isna().sum().any()


# In[16]:


df.head()


# In[17]:


# Preprocessing function
def preprocess_text(text):
    # Tokenize the text into words
    tokens = word_tokenize(text)
    
    # Remove punctuation and convert to lowercase
    tokens = [token.lower() for token in tokens if token.isalpha()]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
   
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join the tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text


# In[18]:


df['text'] = df['text'].apply(preprocess_text)


# In[19]:


df.head()


# # Visualising the data

# In[20]:


# Visualize percentage of user names
plt.figure(figsize=(12, 6))
user_name_count = df['user_name'].value_counts()[:20]
total_users = len(df['user_name'])
percentage_user_name = user_name_count / total_users * 100

ax = sns.barplot(x=user_name_count.index, y=percentage_user_name.values, palette='Set3')
plt.title('Top 20 User Names and Their Percentage')
plt.xlabel('User Names')
plt.ylabel('Percentage')
plt.xticks(rotation=90)

# Display percentage directly on each bar
for i, (count, percentage) in enumerate(zip(user_name_count, percentage_user_name)):
    ax.text(i, percentage + 0.1, f'{percentage:.2f}%', ha='center', va='bottom', fontsize=8)

plt.show()


# In[21]:


# Visualize count and percentage of user locations
plt.figure(figsize=(12, 6))
user_location_count = df['user_location'].value_counts()[:20]
percentage_user_location = user_location_count / len(df) * 100

g = sns.barplot(x=user_location_count.index, y=percentage_user_location.values, palette='Set3')
g.set_title('Top 20 User Locations and Their Percentage')
plt.xlabel('User Locations')
plt.ylabel('Percentage')
plt.xticks(rotation=90)

# Display percentage directly on each bar with reduced vertical position
for i, percentage in enumerate(percentage_user_location):
    g.text(i, percentage + 0.2, f'{percentage:.2f}%', ha='center', va='bottom', fontsize=8)

plt.show()


# In[22]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

stopwords = set(STOPWORDS)
stopwords.update(["t", "co", "https", "amp", "U", "Comment", "text", "attr", "object"])

def show_wordcloud(data, title=""):
    text = " ".join(t for t in data.dropna())
    wordcloud = WordCloud(stopwords=stopwords, scale=4, max_font_size=50, max_words=500, background_color="white").generate(text)
    fig = plt.figure(1, figsize=(16, 16))
    plt.axis('off')
    fig.suptitle(title, fontsize=20)
    fig.subplots_adjust(top=2.3)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.show()

# Example usage
show_wordcloud(df['text'], title="Word Cloud for Tweets")


# In[23]:


show_wordcloud(df['text'], title="Prevalent words in tweets")


# In[24]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer

sentiment = SentimentIntensityAnalyzer()

print(sentiment.polarity_scores('This move is great!'))
print(sentiment.polarity_scores('This move is not great'))


# In[25]:


df['compound'] = df['text'].apply(lambda x: sentiment.polarity_scores(x)['compound'])
df['neg'] = df['text'].apply(lambda x: sentiment.polarity_scores(x)['neg'])
df['neu'] = df['text'].apply(lambda x: sentiment.polarity_scores(x)['neu'])
df['pos'] = df['text'].apply(lambda x: sentiment.polarity_scores(x)['pos'])


# In[26]:


df.sample(10)


# In[27]:


df[['compound','neg', 'neu', 'pos']].describe()


# In[28]:


sns.histplot(df['compound'])


# In[29]:


sns.histplot(df['neg'])


# In[30]:


sns.histplot(df['pos'])


# In[31]:


sns.histplot(df['neu'])


# In[32]:


negative_tweet_count = len(df[df['compound'] < 0])

print("Number of negative tweets:", negative_tweet_count)
print("Sample of negative tweets:")
print(df[df['compound'] < 0]['text'].head())


# In[33]:


positive_tweet_count = len(df[df['compound'] > 0])

print("Number of positive tweets:", positive_tweet_count)
print("Sample of positive tweets:")
print(df[df['compound'] > 0]['text'].head())


# In[34]:


neutral_tweet_count = len(df[df['compound'] == 0])

print("Number of neutral tweets:", neutral_tweet_count)
print("Sample of neutral tweets:")
print(df[df['compound'] == 0]['text'].head())


# In[35]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Assuming you have already defined df
negative_tweet_count = len(df[df['compound'] < 0])

# Create a list of tokens from all negative tweets
neg_tokens = [word for tweet in df[df['compound'] < 0]['text'] for word in tweet.split()]

# Generate word cloud
wordcloud = WordCloud(background_color='black').generate_from_text(' '.join(neg_tokens))

# Plot the word cloud
plt.figure(figsize=(12, 12))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[36]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Assuming you have already defined df and preprocess_text
negative_tweet_count = len(df[df['compound'] < 0])

# Concatenate all processed negative tweets into a single string
negative_tweets_text = ' '.join(df[df['compound'] < 0]['text'])

# Generate word cloud
wordcloud = WordCloud(background_color='black').generate_from_text(negative_tweets_text)

# Plot the word cloud
plt.figure(figsize=(15, 15))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[37]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Assuming you have already defined df
positive_tweet_count = len(df[df['compound'] > 0])

# Create a list of tokens from all positive tweets
pos_tokens = [word for tweet in df[df['compound'] > 0]['text'] for word in tweet.split()]

# Generate word cloud
wordcloud = WordCloud(background_color='black').generate_from_text(' '.join(pos_tokens))

# Plot the word cloud
plt.figure(figsize=(12, 12))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[38]:


from nltk.probability import FreqDist

pos_freqdist = FreqDist(pos_tokens)

pos_freqdist.tabulate(10)


# In[39]:


from nltk.probability import FreqDist

neg_freqdist = FreqDist(neg_tokens)

neg_freqdist.tabulate(10)


# In[40]:


pos_freqdist.plot(30)


# In[41]:


neg_freqdist.plot(30)


# In[42]:


import numpy as np

# Create a new column 'text_sentiment' based on compound scores
df['text_sentiment'] = np.where(df['compound'] > 0, 'Positive', 'Negative')

# Display the updated DataFrame
print(df[['text', 'compound', 'text_sentiment']].head())


# In[43]:


# Convert 'Negative' to 0 and 'Positive' to 1 in the 'text_sentiment' column
df['text_sentiment'] = df['text_sentiment'].map({'Negative': 0, 'Positive': 1})

# Display the updated DataFrame
print(df[['text', 'compound', 'text_sentiment']].head())


# In[44]:


sns.countplot(data=df, x='text_sentiment' )


# In[45]:


df.isna().sum().any()


# In[46]:


df


# In[47]:


# using beautiful soup to scrap html elements
from unidecode import unidecode
df['text'] = df['text'].apply(lambda x: BeautifulSoup(x, "lxml").text)
df['text'] = df['text'].apply(unidecode)


# # Cleaning the data

# In[48]:


count_vectorizer = CountVectorizer(stop_words='english', max_features=40000)
df["clean_text"] = df["text"].dropna().apply(lambda x: x.replace('https', ''))
df["clean_text"] = df["clean_text"].dropna().apply(lambda x: x.replace('http', ''))
df["clean_text"] = df["clean_text"].dropna().apply(lambda x: x.replace('amp', ''))
text_sample = df["clean_text"].dropna().values


print('Headline before vectorization: {}'.format(text_sample[10]))

document_term_matrix = count_vectorizer.fit_transform(text_sample)

print('Headline after vectorization: \n{}'.format(document_term_matrix[10]))


# In[49]:


# extract the features using count vectorizer
vectorizer = CountVectorizer(dtype = 'uint8')
df_vectorizer = vectorizer.fit_transform(df['clean_text'])


# In[50]:


x_train, x_test, y_train, y_test = train_test_split(df_vectorizer, df["text_sentiment"], test_size=.20)


# In[51]:


x_train


# In[52]:


def tokenize(text): 
    tweet_tokenizer = TweetTokenizer()
    return tweet_tokenizer.tokenize(text)

def stem(doc):
    return (stemmer.stem(w) for w in analyzer(doc))



vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    ngram_range=(1, 1))


# # Model Application

# In[53]:


kfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
np.random.seed(1)

pipeline_svm = SVC(probability=True, kernel="linear", class_weight="balanced")


grid_svm = GridSearchCV(estimator=pipeline_svm,
                        param_grid={'C': [ 0.1, 1, 10]},
                        cv=kfolds,
                        scoring="roc_auc",
                        verbose=1,
                        n_jobs=-1)

grid_svm.fit(x_train.toarray(), y_train)
grid_svm.score(x_test.toarray(), y_test)


# In[54]:


prediction_svm = grid_svm.predict(x_test.toarray())

print(prediction_svm)


# In[55]:


pipeline_lr = LogisticRegression(solver='liblinear')

grid_lr = GridSearchCV(pipeline_lr,
                       param_grid={'C': [0.01, 0.1, 1]},
                       cv=kfolds,
                       scoring="roc_auc",
                       verbose=1,
                       n_jobs=-1)

grid_lr.fit(x_train, y_train)
score_lr = grid_lr.score(x_test, y_test)
print(score_lr)


# In[56]:


lin_pred = grid_lr.predict(x_test)
print(lin_pred)


# # Classification Report

# In[57]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score


# In[58]:


# Evaluate the performance
print("SVM classification report:")
print("Accuracy:", accuracy_score(y_test, prediction_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, prediction_svm))
print("Classification Report:\n", classification_report(y_test, prediction_svm))


# In[59]:


cm_svm = confusion_matrix(y_test, prediction_svm)


# In[60]:


# Plotting Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.title('Confusion Matrix - SVM Confusion matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[61]:


# Evaluate the performance
print("Logistic Regression report:")
print("Accuracy:", accuracy_score(y_test, lin_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, lin_pred))
print("Classification Report:\n", classification_report(y_test, lin_pred))


# In[62]:


cm_lr = confusion_matrix(y_test, lin_pred)


# In[63]:


# Plotting Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.title('Confusion Matrix - LR Confusion matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[64]:


from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning for SVM
param_grid_svm = {'C': [0.1, 1, 10]}
grid_svm = GridSearchCV(estimator=pipeline_svm,
                        param_grid=param_grid_svm,
                        cv=kfolds,
                        scoring="roc_auc",
                        verbose=1,
                        n_jobs=-1)
grid_svm.fit(x_train.toarray(), y_train)

# Get the best SVM estimator and its score
best_svm = grid_svm.best_estimator_
best_svm_score = grid_svm.best_score_

# Hyperparameter tuning for Logistic Regression
param_grid_lr = {'C': [0.01, 0.1, 1]}
grid_lr = GridSearchCV(pipeline_lr,
                       param_grid=param_grid_lr,
                       cv=kfolds,
                       scoring="roc_auc",
                       verbose=1,
                       n_jobs=-1)
grid_lr.fit(x_train, y_train)

# Get the best Logistic Regression estimator and its score
best_lr = grid_lr.best_estimator_
best_lr_score = grid_lr.best_score_

# Stacking the optimized classifiers
estimators = [('svm', best_svm), ('lr', best_lr)]
stacking_clf = StackingClassifier(estimators=estimators,
                                  final_estimator=LogisticRegression(solver='liblinear'))
stacking_clf.fit(x_train, y_train)

# Evaluate the stacking classifier
score_stacking = stacking_clf.score(x_test, y_test)
print(score_stacking)


# # ROC curve

# In[65]:


def get_roc_curve(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, pred_proba)
    return fpr, tpr


# In[66]:


# svm roc
roc_svm = get_roc_curve(grid_svm.best_estimator_, x_test.toarray(), y_test)
fpr, tpr = roc_svm
plt.figure(figsize=(14,8))
plt.plot(fpr, tpr, color="red")
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM Roc curve')
plt.show()


# In[67]:


# logistic regression roc
roc_log= get_roc_curve(grid_lr.best_estimator_, x_test, y_test)
fpr, tpr = roc_log
plt.figure(figsize=(14,8))
plt.plot(fpr, tpr, color="red")
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression Roc curve')
plt.show()

