# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:57:43 2022

@author: Nasir
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.stem.porter import PorterStemmer
# nltk.download('stopwords')
# nltk.download('webtext')
from nltk.corpus import stopwords,webtext
from nltk.probability import FreqDist
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score,calinski_harabasz_score
import warnings
import nltk
from nltk.corpus import stopwords
warnings.filterwarnings('ignore')


df = pd.read_csv(r"C:\Users\Nasir\Desktop\Data_Science\Resume_classifier_using_ML\resumes_data.csv")
df=df.drop(['Unnamed: 0'], axis = 1)

df.info()


    
df['resume_name']=df['resume_name'].str.replace("[^a-zA-Z#]", " ")
df['resume_name'] = df['resume_name'].apply(lambda x : x.lower())
Term_count=df['resume_name'].str.split(expand=True).stack().value_counts()
Term_count=pd.DataFrame({'Term':Term_count.index, 'Count':Term_count.values})

Row_list =[]
  
# Iterate over each row
for rows in Term_count.itertuples():
    # Create list for the current row
    my_list =rows.Term
    if rows.Count >=9:
    # append the list to the final list
        Row_list.append(my_list)
        
for i in Row_list:
    df['resume_name']=df['resume_name'].str.replace(i, "")
    

df['resume_description'] = df['resume_description'].apply(lambda x : x.lower())
# Remove html_tags

import re
def remove_html_tags(text):
    pattern=re.compile('<,*?>')
    return pattern.sub(r'',text)
df['resume_description']=df['resume_description'].apply(remove_html_tags)
## Remove URL tags

def remove_url(text):
    pattern=re.compile(r'https?://\S+}www\.\S+')
    return pattern.sub(r'',text)
df['resume_description']=df['resume_description'].apply(remove_url)

## removing punctuation 
import string
string.punctuation
exclude = string.punctuation

def remove_punc(text):
       return text.translate(str.maketrans('','',exclude))
   
df['resume_description']=df['resume_description'].apply(remove_punc)

# Removing Stopwords
from nltk.corpus import stopwords
stopwords.words('english')

def remove_stopwords(text):
    new_text=[]
        
    for word in text.split():
        
        if word in stopwords.words('english'):
            new_text.append('')
        else:
            new_text.append(word)
    x=new_text[:]
    new_text.clear()
    return " ".join(x)

df['resume_description']= df['resume_description'].apply(remove_stopwords)

Term_count1=df['resume_description'].str.split(expand=True).stack().value_counts()
Term_count1=pd.DataFrame({'Term':Term_count1.index, 'Count':Term_count1.values})


Row_list =[]
a=[]  
b=[]
# Iterate over each row
for rows in Term_count1.itertuples():
    # Create list for the current row
    my_list =rows.Term
    if rows.Count >5:
    # append the list to the final list
        Row_list.append(my_list)


for i in df['resume_description']:
    i=i.split()
    for j in i:
        if j in Row_list:
            a.append(j)
            
    b.append(a)
    a=[]
    
    
    
d=[]
for i in b:
    c=' '.join(i)
    d.append(c)    
    
    
df['cleaned_resume_description'] = d


vectorizer = CountVectorizer(max_features=2500)
X = vectorizer.fit_transform(df['cleaned_resume_description']).toarray()
X.shape

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


range_n_clusters = [2,3,4]
silhouette_avg = []
for num_clusters in range_n_clusters:
    # initialise kmeans
    k_means = KMeans(n_clusters=num_clusters,random_state=12)
    k_means.fit(X)
    cluster_labels = k_means.labels_
    silhouette_avg.append(silhouette_score(X, cluster_labels))
    
 
 # silhouette score

plt.plot(range_n_clusters,silhouette_avg) 
plt.title('silhouette_score')
plt.show()

# K-Means
cal_score = []
for i in range(2,5):
    k_means = KMeans(n_clusters=i, random_state=1).fit(X)
    labels = k_means.labels_
    cal_score.append(calinski_harabasz_score(X, labels))
    
plt.plot(range_n_clusters,cal_score)
plt.title('calinski_harabasz_score')
plt.show()

clusters_new = KMeans(4, random_state=42)
clusters_new.fit(X)
#Assign clusters to the data set
df['clusterid_new'] = clusters_new.labels_
df['clusterid_new'].unique()




a=df['clusterid_new'].unique()
for i in a:
    print(i)
    test_resume = df[df['clusterid_new']==i]
    test_resume.reset_index(drop=True,inplace=True)
    Term_count2=test_resume['cleaned_resume_description'].str.split(expand=True).stack().value_counts()
    Term_count2=pd.DataFrame({'Term':Term_count2.index, 'Count':Term_count2.values})
    print('the cluster ',i)
    print(Term_count2.head())
    
# 0=workday ,1=sql,2=peoplesoft

    
