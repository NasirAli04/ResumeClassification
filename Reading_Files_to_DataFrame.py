# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 10:23:22 2022

@author: Nasir
"""

import os
import docx2txt
import glob
import pandas as pd  
from pdfminer.high_level import extract_text
import os, subprocess
import re
import string

resume = []
file_name = []

def reading_data(file):
    resume_name = file[file.rfind("/")+1:]
   
    if resume_name.endswith('.docx'):
        try:
            resume.append(docx2txt.process(file))
            file_name.append(resume_name)
        except:
              print(file)
        
    if resume_name.endswith(".pdf"):
        text = extract_text(file)
        resume.append(text)
        file_name.append(resume_name)
        
    if resume_name.endswith(".doc"):
        file = open(file, 'r', encoding="ISO-8859-1")
        text = str(file.read())
        text = re.sub(r'[^\x00-\x7f]',r'', text) 
        text = re.sub('[^a-zA-Z]',' ', text)
        text = ' '.join(word for word in text.split() if len(word)>2)
        resume.append(text)
        file_name.append(resume_name)
        file.close()
        
        

for file in glob.glob(r'C:\Users\Nasir\Desktop\Data_Science\Resume_classifier_using_ML\Data\**\**\*'):
    reading_data(file)
    
for file in glob.glob(r'C:\Users\Nasir\Desktop\Data_Science\Resume_classifier_using_ML\Data\**\*'):
    reading_data(file)
    
resumes_data = pd.DataFrame({'resume_name':file_name,'resume_description':resume})
resumes_data.head()
        
        
