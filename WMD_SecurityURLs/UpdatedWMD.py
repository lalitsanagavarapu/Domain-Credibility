#import all libraries
#pyemd is used for Word movers distance*(earth movers distance)
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
import os.path
import shutil
import pickle
#to save the gigaword model
from gensim.models import KeyedVectors   


#Lists and keywords in the list

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


#create a series for each list
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

#create a parentlist 

Lists =  [Network_s, Application_s, Access_s,Operations_Control_s,Management_s,Endpoint_s,Malware_s,Cloud_s,Hardware_s,Physical_s]  
#soup = bs4.BeautifulSoup(req.text, 'html.parser')

#Convert list into list of lists
def extractlst(lst): 
    return list(map(lambda el:[el], lst))

#nlp - cleantext 
def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text= re.sub('[^a-zA-Z]', ' ', text )
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text
'''
#raises error for status codes like 404,500 
import urllib.request
def getResponseCode(url):
    conn = urllib.request.urlopen(url)
    return conn.getcode()
'''

#import httplib

#def get_status_code(host, path="/"):
    """ This function retreives the status code of a website by requesting
        HEAD data from the host. This means that it only requests the headers.
        If the host cannot be reached or something else goes wrong, it returns
        None instead.
    """
 #   try:
 #       conn = httplib.HTTPConnection(host)
 #       conn.request("HEAD", path)
 #       return conn.getresponse().status
 #   except StandardError:
 #       return None


#print get_status_code("stackoverflow.com") # prints 200
#print get_status_code("stackoverflow.com", "/nonexistant") # prints 404
#Raise an error if an erroneous status code has been returned

import requests
try:
    r = requests.head("https://stackoverflow.com")
    print(r.status_code)
    # prints the int of the status code. Find more at httpstatusrappers.com :)
except requests.ConnectionError:
    print("failed to connect")

random.seed(365)

#create pandas dataframes to-be stored as a csv file 
df_loc_url=pd.DataFrame()
final_df=pd.DataFrame()
df_normalized=pd.DataFrame()

#store lists and data
ss300_word=[]
ss300_list=[]
ss100_word=[]     
timestamp=[]
ss100_list=[]


url_list_append = []
keyword_append=[]
listname_append=[]
time_list=[]

#IF NOT DOWNLOADED ,run savemode.py first(in same directory)
#these lines can be run as a seperate python file
#make sure to import required gensim libraries
#w2v_model_300 = api.load("glove-wiki-gigaword-300")
#w2v_model_100 = api.load("glove-wiki-gigaword-100")

#load pretrained word2vec model(gensim)

w2v_model_300= KeyedVectors.load_word2vec_format("model300.bin", binary=True)
print("Model 300 loaded")
w2v_model_100= KeyedVectors.load_word2vec_format("model100.bin", binary=True)
print("Model 100 loaded")

counter = 0
print("Open file")

#fancy way to set file path

#cur_path = os.path.dirname(__file__)
#new_path = os.path.relpath('../urls/1to100.txt', cur_path)
#curr_dir = os.getcwd() #Gets the current working directory

#open url.txt file to read urls.
#give your choice of file
#add relative path if in cwd or absolute path

with open("_150_C1_urls.txt","r") as f:
	urls=[url.split() for url in f.readlines()]
	start_time=time.clock()
	currlist=urls
	list_to_map=extractlst(currlist)
	print("List of lists is",list_to_map)

	print("Start cleaning")
	#clean urls(remove extra tags)
	for lists in list_to_map:
		for listin in lists:
			for each,file in enumerate(listin,start=1):
				print("{}.{}".format(each,file))
				char=file.replace("u'http","http")
				char=char.replace("[http","http")
				char=char.replace("',"," ")

				#print(char)
				try:
					f=urllib.request.urlopen(char)
					temp_file=f.read()
					clean=cleanText(temp_file)

					print("Cleaned text is ",clean)
				except:
					pass
				#We now enter WMD calculation for our lists and keywords with URLS
				print("WMD calculation")
				cores=multiprocessing.cpu_count()
				#keep track of cpu count
				print('Total cores in use',cores)

				print('Iterate')
				#get each list in Lists(network,physical...etc)
				for sublist in Lists:
					flag=0
					lenlist=len(sublist)
					#to keep track of list 
					currlist=sublist.name
					print("Current in loop",currlist)
					for element in sublist:
                	#print(currlist,lenlist)
						word=element.lower()
						print("List and word and len",currlist,word,lenlist)
						try:
							#store WMD100 and WMD300 for pairs (url,word) and (url,list)
							word300=w2v_model_300.wmdistance(word,clean)
							word100=w2v_model_100.wmdistance(currlist,clean)
							list100=w2v_model_300.wmdistance(word,clean)
							list300=w2v_model_300.wmdistance(currlist,clean)
							print('\n','word300 simcore of ',word,'with url=',char,'is',word300,'\n')
							print('\n','word100 simcore of ',word,'with url=',char,'is',word100,'\n')
							print('\n','list300 simcore of ',currlist,'with url=',char,'is',list100,'\n')
							print('\n','list100 simcore of ',currlist,'with url=',char,'is',list300,'\n')
							ss300_word.append(word300)
							ss300_list.append(list300)
							ss100_word.append(word100)
							ss100_list.append(list100)

							url_list_append.append(char)
							keyword_append.append(word)
							listname_append.append(currlist)
                           
							endtime=time.clock()
							time_taken=endtime-start_time
							time_list.append(time_taken)
							days = datetime.now().strftime("%Y_%m_%d=%H:%M:%S")
							timestamp.append(days)
							lenlist=lenlist-1

							if lenlist==0:
								flag=1
							#if flag==1:
							#	break	

					#if flag==1:
							#break
						except:
							word

print("Completed")
print(len(ss300_word))
print(len(ss300_list))
print(len(ss100_word))
print(len(ss100_list))
print(len(url_list_append))
print(len(keyword_append))
print(len(listname_append))
print(len(time_list))

df_links_url =pd.DataFrame({'url_Name':url_list_append})  


simscore_keyword_frame=pd.DataFrame({'no_of_days': timestamp,'keyword':keyword_append,
    'name_of_list':listname_append,'time_taken':time_list,
    'word_score_100':ss100_word,'list_score_100':ss100_list,
    'word_score_300':ss300_word,'list_score_300':ss300_list})



final_df['time_stamp']=simscore_keyword_frame['no_of_days']
final_df['url']=df_links_url['url_Name']
final_df['keyword']=simscore_keyword_frame['keyword']
final_df['list_name']=simscore_keyword_frame['name_of_list']
final_df['time_taken']=simscore_keyword_frame['time_taken']

final_df['dis_word_score_100']=simscore_keyword_frame['word_score_100']                                    

final_df['dis_list_score_100']=simscore_keyword_frame['list_score_100']

final_df['dis_word_score_300']=simscore_keyword_frame['word_score_300']

final_df['dis_list_score_300']=simscore_keyword_frame['list_score_300']



# Infinity WMD =0 
#float(inf) is only returned for out-of-vocab words)
#If one of the documents have no words that exist in the vocab, 
#float(‘inf’) (i.e. infinity) will be returned.

final_df['dis_word_score_100'] = final_df['dis_word_score_100'].replace(np.inf, 0)
final_df['dis_list_score_100'] = final_df['dis_list_score_100'].replace(np.inf, 0)
final_df['dis_word_score_300'] = final_df['dis_word_score_300'].replace(np.inf, 0)
final_df['dis_list_score_300'] = final_df['dis_list_score_300'].replace(np.inf, 0)


#normalising WMD since it has no upper/lowerbound 
x_100 = final_df['dis_word_score_100']
y_100= final_df['dis_list_score_100']
x_300 = final_df['dis_word_score_300']
y_300= final_df['dis_list_score_300']


df_normalized['dis_word_score_100']=sklearn.preprocessing.minmax_scale(x_100, feature_range=(0, 1), axis=0, copy=True)
df_normalized['dis_list_score_100']=sklearn.preprocessing.minmax_scale(y_100, feature_range=(0, 1), axis=0, copy=True)

df_normalized['dis_word_score_300']=sklearn.preprocessing.minmax_scale(x_300, feature_range=(0, 1), axis=0, copy=True)
df_normalized['dis_list_score_300']=sklearn.preprocessing.minmax_scale(y_300, feature_range=(0, 1), axis=0, copy=True)

final_df['nor_word_sim_score_100']=df_normalized['dis_word_score_100']
final_df['nor_list_sim_score_100']=df_normalized['dis_list_score_100']
final_df['nor_word_sim_score_300']=df_normalized['dis_word_score_300']
final_df['nor_list_sim_score_300']=df_normalized['dis_list_score_300']
#returns N rows (2 here) , this is just to check correctness of dataframe being created
print("Test pandas",final_df.head(2))

_150_C1_urls=final_df.to_csv("/home/abhishek/Avi/SimScore/CSV/_150_C1_urls.csv", header=True)


#TO-DO

#the following code can be spedup if we identify a pattern in similairty scores 
#it may be possible that we might only need to check list similarity with the url 
#can we use multithreading?
