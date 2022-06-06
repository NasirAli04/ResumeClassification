# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:35:29 2022

@author: Nasir
"""
import os
import docx2txt
import glob
from pyresparser import ResumeParser #Library 2 
import pandas as pd       

text = ''
skills_list=[]
for file in glob.glob('Data\*.docx'):
    text += docx2txt.process(file)

    
    try:
        file.remove('word/document.xml')

    except: 
        pass
    

os.chdir(r'C:\Users\Nasir\Desktop\Data_Science\New folder')

Descriptions = []
name = []

for file in glob.glob('**\\*.docx'):
    Descriptions.append(docx2txt.process(file))    
    name.append(file)
    skills = ResumeParser(file).get_extracted_data()
    skills_list.append(skills['skills'])

data = pd.DataFrame(
    {'Descriptions': Descriptions,
     'Name': name,
     "Skils":skills_list
     
    })

for file in glob.glob('**\\**\\*.docx'):
    Descriptions.append(docx2txt.process(file))    
    name.append(file)
    skills = ResumeParser(file).get_extracted_data()
    skills_list.append(skills['skills'])

data1 = pd.DataFrame(
    {'Descriptions': Descriptions,
     'Name': name,
     "Skils":skills_list
    })
data.dtypes
df=data1
####################################################################################################################
df.to_csv('all_resumes.csv')
category1=[]
category2=[]
for i in df['Skils']:
    if ('Peoplesoft' and 'Servers' and 'Security' and 'Erp' and 'Analytics' and 'Programming' in i):
        category_name='Peoplesoft'
        value=1   
    #elif ('Sql' in i or 'sql' in i and 'Database' in i  and 'hadoop' and 'Cloud' in i and 'Testing' in i and 'System' 
    #      and 'Technical' and 'Windows' in i and 'Linux' and 'Transactions' in i and  'Troubleshooting' in i ):
    elif ('sql' in i or 'SQL' in i ):
        category_name='sql_developer/tester'
        value=2
    elif ('Reactjs' in i or 'react js' in i)  :
        category_name='ReactJs Developer'
        value=3
    elif ('Analytics' in i or 'Data Analysis' in i) :
        category_name='Data Analytics'
        value=4
    elif ( 'Html' in i and 'CSS' in i or 'Java')  :
        category_name='Software Developer'
        value=5
    else:
        category_name='others'
        value=6
        
    category1.append(category_name)
    category2.append(value)

#print(category)
df['Job_Title']=category1
df['Category']=category2


df
## EDA

# 1.Lower case

df['Clean_Resume'] = df['Descriptions'].apply(lambda x : x.lower())
# Remove html_tags

import re
def remove_html_tags(text):
    pattern=re.compile('<,*?>')
    return pattern.sub(r'',text)
df['Clean_Resume']=df['Clean_Resume'].apply(remove_html_tags)
## Remove URL tags

def remove_url(text):
    pattern=re.compile(r'https?://\S+}www\.\S+')
    return pattern.sub(r'',text)
df['Clean_Resume']=df['Clean_Resume'].apply(remove_url)

## removing punctuation 
import string
string.punctuation
exclude = string.punctuation

def remove_punc(text):
       return text.translate(str.maketrans('','',exclude))
   
df['Clean_Resume']=df['Clean_Resume'].apply(remove_punc)

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

df['Clean_Resume']= df['Clean_Resume'].apply(remove_stopwords)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1000)
bow=cv.fit_transform(df['Clean_Resume'])
x= bow.toarray()
y=df[['Category']]
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score,accuracy_score,confusion_matrix,classification_report

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=1)
x_train.shape,y_train.shape,x_test.shape,y_test.shape
model_nomial=MultinomialNB(alpha=1)
model_gaussian=GaussianNB()
grad=GradientBoostingClassifier(n_estimators=10,random_state=10)
ada=AdaBoostClassifier(n_estimators=100,random_state=10)
extra=ExtraTreesClassifier(n_estimators=100,random_state=10)
random=RandomForestClassifier(n_estimators=100,random_state=10,max_depth=10)
bag=BaggingClassifier(n_estimators=50,random_state=10)
xgb=XGBClassifier(n_estimators=50,random_state=10)
svc=SVC(kernel='linear',gamma=1.0)
KN=KNeighborsClassifier(n_neighbors=10)

clfs= {
    'Multinomial naive byes':model_nomial,
    'Gaussian naive byes':model_gaussian,
    'Extra':extra,
    'Random Forest':random,
    'Bagging':bag,
    'Xtreme Gradient Boosting':xgb,
    'Support Vector Machine':svc,
    'KN':KN
}

def Findallvalues(clf,x_train,y_train,x_test,y_test):
    clf.fit(x_train,y_train)
    y_predict=clf.predict(x_test)
    accuracy = accuracy_score(y_test,y_predict)
    precision = precision_score(y_test,y_predict,pos_label='positive',average='micro')
    
    return accuracy,precision

accuracy_scores=[]
precision_scores=[]


for name,clf in clfs.items():
    
    current_accuracy,current_precision= Findallvalues(clf,x_train,y_train,x_test,y_test)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)
    
performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores}).sort_values('Accuracy',ascending=False)
performance_df

bag.fit(x_train,y_train)
y_predict_test_bag=bag.predict(x_test)

print("Classification Report \n",classification_report(y_test,y_predict_test_bag))
print("Precision Score \n",precision_score(y_test,y_predict_test_bag,pos_label='positive',average='micro'))

from pickle import load,dump
#Pickle file 
dump(xgb,open('Xtreme_Gradient_Boosting.pkl','wb'))
#------------------------------------------------------------------------------------------------------------------
model = load(open('Xtreme_Gradient_Boosting.pkl','rb'))
y=model.predict(x_test)

