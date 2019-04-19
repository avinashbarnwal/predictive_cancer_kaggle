import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import nltk
import os


path=''
os.chdir(path)

token_dict = {}
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def containsNonAscii(s):
    return any(ord(i)>127 for i in s)

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    #tokens=[word.lower() for word in tokens if word.isalpha()]
    tokens=[word.lower() for word in tokens if not containsNonAscii(word) and word.isalpha()]
    #print(tokens)
    stems = stem_tokens(tokens, stemmer)
    return stems

def binning(Var):
	


data_training = pd.read_csv("training_variants.csv",sep=",")
print(data_training.head())

file = open("training_text.csv", "rU")
file.readline() #ignore first line
no_id = sum(1 for line in open("training_text.csv","rU"))-1
print(no_id)
data_text = pd.DataFrame()
counter = 0
for line in file:
    data_text = data_text.append({'ID': line.split("||")[0] , 'TEXT': line.split("||")[1]}, ignore_index=True)

print(data_text.head())


cv = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
x_traincv = cv.fit_transform(data_text['TEXT'])


a = x_traincv.toarray()
features = np.array(cv.get_feature_names())
#features = pd.DataFrame(features)
print(features)






data_training = pd.read_csv("training_variants.csv",sep=",")
print(data_training.head())

file = open("training_text.csv", "rU")
file.readline() #ignore first line
no_id = sum(1 for line in open("training_text.csv","rU"))-1
print(no_id)
data_text = pd.DataFrame()
counter = 0
for line in file:
    data_text = data_text.append({'ID': line.split("||")[0] , 'TEXT': line.split("||")[1]}, ignore_index=True)

print(data_text.head())


data_text['ID'] = data_text['ID'].astype('int')
data_text = pd.merge(data_text,data_training,how='inner', on=['ID'])
print(data_text['Class'])

def top_tfidf_feats(row, features, top_n=25):
    #''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids  = np.argsort(row)[::-1][:top_n]
    print(topn_ids)
    print(topn_ids[1])
    print(features[topn_ids])
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_feats_in_doc(Xtr, features, row_id, top_n=25):
#Top tfidf features in specific document (matrix row)
    row = np.squeeze(Xtr[row_id])
    return top_tfidf_feats(row, features, top_n)

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    #''' Return the top n features that on average are most important amongst documents in rows
    #    indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids]
    else:
        D = Xtr

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)


def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=25):
    #''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
    #    calculated across documents with the same class label. '''
    dfs = pd.DataFrame()
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        no = len(feats_df['feature'])
        feats_df['label'] = np.repeat(label,no)
        #print(feats_df)
        dfs = dfs.append(feats_df)
        print(dfs)
    return dfs

result = top_feats_by_class(a,data_text['Class'],features,min_tfidf=0.1,top_n=25)
#print(result)
result.to_csv('result.csv',sep=",",encoding = 'utf-8')




selected_features = result['feature'].unique()
selected_features = u' '.join(selected_features).encode('utf-8').strip()
selected_features = selected_features.split()


features = np.array(cv.get_feature_names())
features = u' '.join(features).encode('utf-8').strip()
features = np.array(features.split())

features_space = pd.DataFrame(a)
features_space.columns = features
selected_feature_space = features_space[selected_features])

