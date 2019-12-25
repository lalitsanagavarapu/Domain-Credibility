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

30/11/19
- 
1.For a security search engine specific task, and since this is credibility assessment<br>
-- Inverse Reinforcement learning. (targetted recommendations) <br>
-- BERT (Used by google for youtube searches) <br>
2. Check Zenserp api<br>
3.What should be the search bias? <br> 
4.MultiLabel genres? <br>

1/12/19
-
1.Setup the environment to run existing code. <br>
2.Postgres DB<br>
3.WebCred-dev Up and running (always gives default values as output, timeouts for some urls and exceptions as well)
4.Need to fix WebcCred-dev.
<br>


2/12/19
--
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
8.[Webcred](https://github.com/SIREN-SERC/WEBCred)



4/12/19
-
1.Doc2Vec or Word2Vec <br>
2.WDM better for words which have nothing in common. <br>
3.Implemented BFS/DFS for finding similar words in the knowledge graph for the given list of words/word<br>
4.Research more on which approach would be better ? Do we need to implement KdTrees? <br>
5.Glove/Gensim<br>
6.Find all child urls of a given url.<br>

5/12/19
- 
1.Looked up page ranking<br>
2.Looked into Facebook FAISS<br>
3.Find advanced ranking algorithms<br>

6/12/19
-
1.Which model to use ?<br>
2.Implemented Sentence Similarity using wordnet based on a paper.<br>
3.TO-DO: Update existing code to python3.x <br>
4.[Webcred-dev](https://github.com/SIREN-SERC/WEBCred-dev) still has some issues to be resolved.<br>

9/12/19
-
1.Implemented Textranking for keyword extraction<br>
2.Understood POS Tagging,Lemmatization,stopword removal and vocabulary creation.Built a knowledge graph from scratch for the vocabulary created.
Assigned scores to vertices. <br>
3.Ranked keyphrases. The above implementation was completely based on a paper cited<br>.

10/12/19
-
1.Implemented basic searching of a keyword in security ontolgy.<br>
2.Crawled contenst of a website to get similar phrases for input keyword.<br>
3.Implementing BFS DFS to search for most similar URL's in the knowledge graph.<br>
4.Adding keyword search functionality to webcred<br>
5.Need to update check_genre function in [Webcred-dev](https://github.com/SIREN-SERC/WEBCred-dev) {currently nothing implemented}.<br>
6.Working on WebCred-plugin<br>


11/12/19
-
1.Working on Plugin.<br>
2.Credibility score not being returned in existing code.<br>
3.Need to add credibility score functionality and integrate with plugin.<br> 
4.Recaptcha not working,genereated new keys.(put aside for the moment).<br>
5.Frontend of keyword searching implemented on current webcred server.<br> 



13/12/19
-
#### Code pushed
1.Keyword based url retrieval ranked by cosine similarity between the query word and the list of urls in the security.owl file.<br>
2.BFS/DFS :Results for both searching algorithms have been stored.<br>
3.Urls are ranked in descending order and returned as a list with the relevance score for each.<br>
4.Working on generate score function for the plugin,working on backend.<br>
5.Need to figure out the check_genre function.<br> 


16/12/19
-
1.WEBCred assigning null to calculated values. Fixing that.<br>
2.check_genre function needs an reponse from the ML model.Working on that.<br>

17/12/19
- 
1.ML model<br>
2.Similarity scores still running(taking time..)<br>
3.Wrote backend for plugin<br>
4.URL extracter and web crawler fixed<br>

19/12/19
- 
1.Implementing ML model ,dataset for training getting ready<br>
2.Similarity score Updated and fixed, CSV files now make sense<br>

20/12/19
- 
1.WMD.py was producing redundant results,issue fixed.Code has been optimized and made more efficient<br>
2.Need to handle exceptions for URL's which sent back error status codes<br>
3.Plugin is now working(please run it on google chrome due to CORS error in firefox)<br>

23/12/19
-
1.Plugin needs to be linked to original Webcred score script.<br>
2.Generated CSV files for 700 URL's in batches, it takes about 1 hour for 150 urls (which can be sped up since multiple tasks were running in the machine)<br>
3.Complexity of WMD.py has been reduced by O(len(keywordlist))<br>

24/12/19
-
1.TF-IDF can be smoothed<br>.
2.Apache SOLR<br> (look up)<br>

25/12/19
-
1.WMD.py made more efficient, only parses relevant webpage content,excludes blacklist.(style,headers like html tags)<br>
2.GBDT to be run using our generated dataset.<br>
