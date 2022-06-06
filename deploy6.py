# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 14:22:01 2022

@author: Nasir
"""

import streamlit as st
import pandas as pd 
import docx2txt
from PyPDF2 import PdfFileReader
import pdfplumber
import re 
import nltk 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
from pickle import load,dump
import win32com.client as win32
from win32com.client import constants


def Read_pdf(file):
    pdfreader=PdfFileReader(file)
    count=pdfreader.numPages
    all_page_text=""
    for i in range(count):
        page=pdfreader.getPage(i)
        all_page_text += page.extractText()
    return all_page_text

def Read_docx(file):
    pdraw_text2 = docx2txt.process(file)
    a=[pdraw_text2]
    return a
  
    
   
import numpy as np
c = []

def main():
    st.title("P112 Group_4 Resume Classifier")
    menu=["Home","Documentfile",'About']
    choice = st.sidebar.selectbox("Menu",menu)

    if choice=="Home":
        st.header("Welcome to Resune Classifier")
        image="https://miro.medium.com/max/1200/0*XzaHayyew8qMDRIt.gif"
        st.image(image)
        
    elif choice=="About":
        st.header("Welcome to Resune Classifier by Group 4")
        st.subheader("Mohammad Nasir Ali")
        st.subheader("Manichandana Jillepalli")
        st.subheader("Karthik Ongoji")
        st.subheader("Pal Lucky Kumar Vindhvsin")
        st.subheader("Pooja Pandey")
        st.subheader("Deeban VM")
                         
                	 
        
        
        
    elif choice=="Documentfile":
        docx_file = st.file_uploader("Upload File",type=['txt','docx','pdf','doc'],accept_multiple_files=True)
        for i in docx_file:
            
            if i is not None:
                file_details = {"Filename":i.name,"FileType":i.type,"FileSize":i.size}
                #x`x``st.write(file_details)
                    
                    
                    
                if i.type=="application/pdf":
                    pdf_file=Read_pdf(i)
                    #st.write(pdf_file)
                        
                    
                        
                elif i.type=="application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    do_file=Read_docx(i)
                    b=do_file
                    c.append(b)
                  
                    
                    data1 = pd.DataFrame(
                        {'Text':c,
                         
                        
                         
                         
                         })
                    
               
                   
                    def try_join(l):
                        try:
                            return ','.join(map(str, l))
                        except TypeError:
                            return np.nan

                    data1["Text"] = [try_join(l) for l in data1["Text"]]
                   
                    data1["Text"] = data1["Text"].apply(lambda x : x.lower())
                    
                   
                    # Remove html_tags

                    import re
                    def remove_html_tags(text):
                        pattern=re.compile('<,*?>')
                        return pattern.sub(r'',text)
                    data1["Text"]=data1["Text"].apply(remove_html_tags)
                    ## Remove URL tags

                    def remove_url(text):
                        pattern=re.compile(r'https?://\S+}www\.\S+')
                        return pattern.sub(r'',text)
                    data1["Text"]= data1["Text"].apply(remove_url)

                    ## removing punctuation 
                    import string
                    
                    exclude = string.punctuation

                    def remove_punc(text):
                           return text.translate(str.maketrans('','',exclude))
                       
                    data1["Text"]=data1["Text"].apply(remove_punc)

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
                e=''
                f=''
                data1["Text"]=  data1["Text"].apply(remove_stopwords)
                
                    
        from sklearn.feature_extraction.text import CountVectorizer
        cv=CountVectorizer(max_features=1000)
        bow=cv.fit_transform( data1["Text"])
        x= bow.toarray()
        model = load(open('NBmodela.pkl','rb'))
        y=model.predict(x)
        
        
        df = pd.DataFrame(y, columns = ['prediction'])
       
        
        
        df['prediction']=df['prediction'].map({1:'Peoplesoft',2:'REACT_Developer',3:'WORKDAY',4:'SQL',0:'Software developer'})
        st.write(df)
            
            
            
        
        
        
        
                   # Remove html_tags
                # Remove html_tags
              

   
                    
                

                   
                    
                        
                    
                    
                        
                    
                    
        

                    
                    
                   
                        
                    
                    
                    
               
                    
#------------------------------------------------------------------------------------------------------------------
                    
                 
                    
                      
        
if __name__ == '__main__':
	main()
    


        
