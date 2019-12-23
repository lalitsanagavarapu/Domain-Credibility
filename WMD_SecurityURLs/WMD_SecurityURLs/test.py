import pyemd
import gensim
from gensim import models
from gensim import corpora
from gensim.models.doc2vec import Doc2Vec
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import numpy as np
import urllib
import pandas as pd
import requests
import io
from bs4 import BeautifulSoup
import time
import bs4 as bs
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import math
#import sklearn


# In[3]:


#from fake_useragent import UserAgent
from gensim.models.doc2vec import TaggedDocument
#from sklearn.metrics import accuracy_score
import os
#from sklearn import svm
from nltk.tokenize import sent_tokenize
from random import shuffle
#from sklearn.linear_model import LogisticRegression
from gensim.test.utils import common_texts
from gensim.corpora import Dictionary
from gensim.models import Word2Vec, WordEmbeddingSimilarityIndex
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix
from gensim.similarities import WmdSimilarity
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedLineDocument
import random
from nltk.corpus import stopwords
import glob
import codecs
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
#from sklearn.datasets import fetch_20newsgroups          
from nltk import download
import socket
from socket import *
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pprint import pprint
import multiprocessing
import gensim.downloader as api
import sklearn
from sklearn import preprocessing
from datetime import datetime


# In[ ]:





# In[4]:


Network= ["Firewall","Rules","Port","OS","operating system","Management Console","Throughput","Intrusion Prevention System",
"Intrusion Detection System","Network Time Protocol","Virtual Private Network","LAN","local area network","wide area network","WAN",
"virtual local area network","virtual LAN","VLAN","Wireless","Internet","Switch","Router","MUX","multiplex"]
Application = ["Operating System","Data","Database","File","Physical","Shred","Soft","Stationery","Flight","Transfer","Web","Logs"
"Middle Tier","Logs"]
Access=["Account","Directory Services","Authorization","Authentication","Cryptography"]
OperationsControl=["Network","Security","Monitor","Asset" "Management","Classification","Tag","Audit","Audit",
                    "Incident","Threat","Vulnerability","Attack","Severity","Risk","Impact","Probability","Root cause analysis"]
Management=["Policy","People","Recruit","Employee","Back Ground Verification","Vendor","Engage","Training","Awareness","Disciplinary",
"Exit","Remote" , "Telework","Outsider","Standard","Procedure","Governance","Reporting","Review","Change Control","Patch",
"Contract","SLA","service level agreement","NDA","non disclosure agreement","Law", "IPR","Intellectual property","Compliance","Regulation",
"BCP","Business Continuity Plan ","DR","Disaster Recovery","Backup","Data",
            "Application","Site","Load Balancing","Redundant","People","Call tree"]
Endpoint=["Computer","Laptop","Thin Client","Mobile Device","Removable Media","Pen drive","CD","DVD"]
Malware=["Anti-virus","Signature","Phishing","Smishing","Vishing","CSRF","Cross-site request forgery"]
Cloud=["IaaS","PaaS","SaaS","Software as a Service ","Platform as a Service","Infrastructure as a Service "]
Hardware=["Server","Chassis","Blade","Controller","Rack mounted","Tower","Storage","SAN","storage area network","Controller","NAS",
"Network-attached storage","Controller","Disks","SSD","Solid State Drive","Magnetic","Tape"]
Physical=["Desk","Clean" ,"Clear","Access","Card"
,"Password","one-time password ","OTP","Biometric","Log","Retina","Fire Extinguisher","Gas","Liquid","Solid","Emergency",
          "Area","Door","Window","Lightning resister","Lock","Door","Office","Power",
          "Uninterruptable Power Supply","Diesel Generator","Electricity","Surge Protector","Location",
          "Altitude","Latitude","Loading","Delivery","Surveillance","Temparature","Air","Video","Monitoring","Building",
          "Asset","HVAC","Heating ventilation and air conditioning","Air conditioning","Precision","Chilled","Ventillation","Heating","Alarm","Fire","Rodent","Water",
          "Smoke","Intrusion","Floor","Raised","Ceiling","False","Rack"]


# In[5]:


Network_s = pd.Series(Network,name='Network')
Application_s = pd.Series(Application,name='Application')
Access_s = pd.Series(Access,name='Access')
Operations_Control_s = pd.Series(OperationsControl,name='OperationsControl')
Management_s = pd.Series(Management,name='Management')
Endpoint_s = pd.Series(Physical,name='Endpoint')
Malware_s = pd.Series(Malware,name='Malware')
Cloud_s = pd.Series(Cloud,name='Cloud')
Hardware_s = pd.Series(Hardware,name='Hardware')
Physical_s = pd.Series(Physical,name='Physical')


# In[6]:


#Lists =  [Network_s]
Lists =  [Network_s, Application_s, Access_s,Operations_Control_s,Management_s,Endpoint_s,Malware_s,Cloud_s,Hardware_s,Physical_s]  
#soup = bs4.BeautifulSoup(req.text, 'html.parser')
def extractlst(lst): 
    return list(map(lambda el:[el], lst))
def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text= re.sub('[^a-zA-Z]', ' ', text )
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text


random.seed(365)

df_loc_url2=pd.DataFrame()
final_df2=pd.DataFrame()
df_normalized=pd.DataFrame()


ss300_word=[]
ss300_list=[]
ss100_word=[]     


ss100_list=[]


url_list_append = []
keyword_append=[]
listname_append=[]
time_list=[]


w2v_model_300 = api.load("glove-wiki-gigaword-300")
print("model 300 tained")
w2v_model_100 = api.load("glove-wiki-gigaword-100")
print("model 100 trained")

counter = 0
flag=0
print("Start URL extraction")
with open("url5.txt", "r") as f:
    #start_urls = [url.strip() for url in f.readlines()]
        
        start_urls = [url.split() for url in f.readlines()]
        #working##print(start_urls)
        start_time = time.clock()
        lst =start_urls
        exlist=extractlst(lst)
        print(exlist)

        print("Start cleaning")
        for ul  in  exlist:
            for uull in ul:
                #encodeline=uull.encode("utf-8")

                for each,file  in enumerate(uull,start=1):
                    print ("{}.{}".format(each,file))
                    #l = [str(i) for i in file]
                    char = file.replace("u'http","http")
                    char = char.replace("[http","http")
                    char = char.replace("',"," ")
                    #char = file.replace("*http://","http://")

                  
                    print(char)
                    try:
                        f = urllib.request.urlopen(char)
                        file1 = f.read()
                        cleaned_txt_2=cleanText(file1)
                       #l = [item for value in file for item in value]
                    #l = ''.join(l)
                    ##Find(file)
                        print("cleaned_txt")
                    except:
                        pass
                    print('Start the model')
                    cores = multiprocessing.cpu_count()
                    print('Total Cores being used',cores)

           
                    print('Iterate')
                    for sublist in Lists:
                        lenlist=len(sublist)
                        #lower_list = [element.lower() for element in sublist]
                        k=sublist.name

                        print("Current list is ",k)
                        #flag=1
                        for element in sublist:
                            lower_element = [element.lower() for element in sublist]
                            for wrd in  lower_element:
                            #wrd=lower_element
                                #print(wrd)
                                    try:
                                            # word_vectors_model.init_sims(replace=True)
                                            sim_word_300= w2v_model_300.wmdistance( wrd,cleaned_txt_2)
                                            sim_list_300=w2v_model_300.wmdistance( k ,cleaned_txt_2)
                                            sim_word_100= w2v_model_100.wmdistance( wrd,cleaned_txt_2)
                                            sim_list_100=w2v_model_100.wmdistance( k ,cleaned_txt_2)
                                            print('\n', ' wrd similarity_score of ',wrd,"  with url" ,char,'  is  ',sim_word_300,'\n')
                                            print('\n', ' list similarity_score of ',k,"  with url" ,char,'  is  ',sim_list_300,'\n')
                                            print('\n', ' wrd similarity_score of ',wrd,"  with url" ,char,'  is  ',sim_word_100,'\n')
                                            print('\n', ' list similarity_score of ',k,"  with url" ,char,'  is  ',sim_list_100,'\n')
                                            ss300_word.append(sim_word_300)
                                            ss300_list.append(sim_list_300)
                                            ss100_word.append(sim_word_100)                     
                                            ss100_list.append(sim_list_100)
                                            url_list_append.append(char)
                                            keyword_append.append(wrd)
                                            listname_append.append(k)

                                            endtime = time.clock()
                                            time_taken = endtime - start_time
                                            time_list.append(time_taken)
                                            lenlist=lenlist-1
                                            
                                            if lenlist==0:
                                                flag=1
                                                break
                                        #if flag==1:
                                         #   break                                        
                                    except:
                                            wrd
                                #if lenlist==0:
                                 #   break
counter += 1 # This line added
print(" done ")  


print(len(ss300_word))
print(len(ss300_list))
print(len(ss100_word))
print(len(ss100_list))
print(len(url_list_append))
print(len(keyword_append))
print(len(listname_append))
print(len(time_list))


# In[ ]:


df_links_url2 =pd.DataFrame({'url_Name':url_list_append})  


# In[ ]:


ss_kw2=pd.DataFrame({'word_score_100':ss100_word,'time_taken':time_list,
    'list_score_100':ss100_list,'word_score_300':ss300_word,'list_score_300':ss300_list,
    'keyword':keyword_append,'name_of_list':listname_append}) 


# In[ ]:


final_df2['url']=df_links_url2['url_Name']
final_df2['time_taken']=ss_kw2['time_taken']
final_df2['list_name']=ss_kw2['name_of_list']
final_df2['keyword']=ss_kw2['keyword']

final_df2['dis_word_score_100']=ss_kw2['word_score_100']
final_df2['dis_list_score_100']=ss_kw2['list_score_100']
final_df2['dis_word_score_300']=ss_kw2['word_score_300']
final_df2['dis_list_score_300']=ss_kw2['list_score_300']



# In[ ]:


final_df2['dis_word_score_100'] = final_df2['dis_word_score_100'].replace(np.inf, 0)
final_df2['dis_list_score_100'] = final_df2['dis_list_score_100'].replace(np.inf, 0)
final_df2['dis_word_score_300'] = final_df2['dis_word_score_300'].replace(np.inf, 0)
final_df2['dis_list_score_300'] = final_df2['dis_list_score_300'].replace(np.inf, 0)


# In[ ]:


x_100 = final_df2['dis_word_score_100']
y_100= final_df2['dis_list_score_100']
x_300 = final_df2['dis_word_score_300']
y_300= final_df2['dis_list_score_300']


# In[ ]:


df_normalized['dis_word_score_100']=sklearn.preprocessing.minmax_scale(x_100, feature_range=(0, 1), axis=0, copy=True)
df_normalized['dis_list_score_100']=sklearn.preprocessing.minmax_scale(y_100, feature_range=(0, 1), axis=0, copy=True)


# In[ ]:


df_normalized['dis_word_score_300']=sklearn.preprocessing.minmax_scale(x_300, feature_range=(0, 1), axis=0, copy=True)
df_normalized['dis_list_score_300']=sklearn.preprocessing.minmax_scale(y_300, feature_range=(0, 1), axis=0, copy=True)


# In[ ]:


final_df2['nor_word_sim_score_100']=df_normalized['dis_word_score_100']
final_df2['nor_list_sim_score_100']=df_normalized['dis_list_score_100']
final_df2['nor_word_sim_score_300']=df_normalized['dis_word_score_300']
final_df2['nor_list_sim_score_300']=df_normalized['dis_list_score_300']
final_df2.head(2)                              
test_score=final_df2.to_csv ("/home/abhishek/Avi/SimScore/test_score.csv", header=True)

