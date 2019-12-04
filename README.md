# WebCred-Api
Currently adopted method is the Gradient boosted decision tree.(XGBoost/LightGBM)  
This works well with tabular datasets.<br>
Features were orthogonal.<br>
Anova was used over mutual gain for feature reduction.



### Dependencies: 
NLTK and BeautifulSoup.<br>
EasyList.<br>
Yslow with PhantomJS <br>
Merceine API <br> 
WebArchives API <br>
StanfordCoreNLP Library<br> 
URLlib<br> 




#### Ideas ::
28/11/19
-------
1.Siamese Net (pairwise comparision, how to design triplet loss) <br>
2.Relevancy Sorting (Used by google, fine-tuned mix of a lot of stuff)<br>
3.For a security search engine specific task, and since this is credibility assessment<br>
-- Inverse Reinforcement learning. (targetted recommendations) <br>
-- BERT (Used by google for youtube searches) <br>
4. Check Zenserp api<br>
5.What should be the search bias? <br> 
6.MultiLabel genres? <br>

2/12/19
--
1.Current credibility is based on URL. <br> 
Issues:
-
1.Fixing WebCred repository,updating doc.<br> 
2.Peer connection closing for some websites (timeout)<br> 
3.Database on remote machine, to be made usable on local machine.<br> 

3/12/19
-
Task:
-
1.Generate word similarity scores, search for query word in knowledge graph and return pages ranked in order of this score( another aspect of credibility for query words/sentences).<br> 
2.Gensim Word2Vec/Doc2Vec<br> 
3.Which searching algorithm to use? (or can we just brute force?)<br> 
4.TF-IDF used by SOLR <br> 
5.Can the knowledge graph be a KDTree(get nearest neighbour words of a new vector)<br> 
6.Reference ontology on security. <br> 
7.Api for the above , plugin development .<br> 










 





