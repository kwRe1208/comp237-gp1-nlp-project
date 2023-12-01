# %%
import pandas as pd

#Assigning the data file to a variable
file_path = r"C:\git\comp237-gp1-nlp-project\dataset\Youtube05-Shakira.csv"

# Replace 'file_path' with the actual path to your data file
df = pd.read_csv(file_path)


# %%
#Displaying the first 5 rows of the data to see what it looks like
df.head(5)

# %%
#Confirming the data type of each column and no missing values
df.info()

# %%
#After reading the data file, we can determine that only the 'CONTENT' and 'CLASS' columns are needed for the analysis.
#We can drop the other columns using the drop() method.
df_filtered = df.drop(['COMMENT_ID', 'AUTHOR', 'DATE'], axis=1)

# %%
#find How many labels of CLASS column
df_filtered['CLASS'].unique()

# %%
# View data with CLASS = 1
df_filtered[df_filtered['CLASS'] == 1].head(5)

# %%
# View data with CLASS = 0
df_filtered[df_filtered['CLASS'] == 0].head(5)

# %% [markdown]
# From above data, we can determine CLASS labeled with 1 is Spam, 0 is non-spam.

# %%
# Shuffle the dataset
df_shuffled = df_filtered.sample(frac=1)

# Print the shuffled dataset
df_shuffled.head(4)


# %%
# Split the dataset into training and test sets with an 75/25 split
X = df_shuffled.drop('CLASS', axis=1) 
y = df_shuffled['CLASS']

train_size = int(len(X) * 0.75) 
test_size = len(X) - train_size

X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# %%
#Build a count vectorizer and extract term counts 
#Building a Category text predictor. 
#Use count_vectorizer.fit_transform().
from sklearn.feature_extraction.text import CountVectorizer

# Create an instance of CountVectorizer
count_vectorizer = CountVectorizer()
train_tc = count_vectorizer.fit_transform(X_train['CONTENT'])
train_tc.shape


# %%
#This downscaling is called tf–idf for “Term Frequency times Inverse Document Frequency”.
# Create the tf-idf transformer
from sklearn.feature_extraction.text import TfidfTransformer

# Create an instance of TfidfTransformer
tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_tc)


# Print the shape of the downscaled data
print("Shape of the downscaled data:", train_tfidf.shape)

# Print any other useful information about the downscaled data
print("Type of the data:", type(train_tfidf))



# %%
from sklearn.naive_bayes import MultinomialNB

# tfidf used for text classification and information retrieval
# Create an instance of the MultinomialNB classifier
classifier = MultinomialNB().fit(train_tfidf, y_train)


# %%
#Using nltk toolkit classes and methods prepare the data for model building
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# %%
from sklearn.metrics import accuracy_score

ypred = classifier.predict(train_tfidf)
accuracy_score(y_train, ypred)

# %% [markdown]
# # Define test data here

# %%
input_data = X_test['CONTENT']

# %%
#Custom input data here

# Create 4 non-spam comments and 2 spam comments
input_data = [
    'This is a great video',
    'I love this video',
    'This video is the best',
    'This video is the worst',
    'Check out my video',
    'Yo must see Snoop Dogg new video!',
    'It is similar to the video I saw yesterday',
    'It is similar to the video with snoop dog'
]


# %%
input_tc = count_vectorizer.transform(input_data)

# %%
input_tfidf = tfidf_transformer.transform(input_tc)
predictions = classifier.predict(input_tfidf)

category_map = {
    0: 'Not Spam', 
    1: 'Spam', 
}

# Print the classification results
for sent, category in zip(input_data, predictions):
    print('\nInput:', sent, '\nPredicted category:', category_map[category])

# %%
#Check the accuracy_score with test data set (0.25)
accuracy_score(y_test, predictions)


