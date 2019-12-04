Original code(depriacated for python3)
[Link](http://sujitpal.blogspot.com/2014/12/semantic-similarity-for-short-sentences.html)


Code updated to run in python3 <br>
Datasets added in the code to see results<br>
Implementation of the paper Sentence Similarity Based on Semantic Nets and corpus statistics.pdf<br>
Can be used in webcred api for ranking webpages based on query word<br>


## TO-DO

Word Mover's Distance is a promising new tool in machine learning that allows us to 
submit a query and return the most relevant documents<br>
Will be explored .<br>
WMD is a method that allows us to assess the "distance" between two documents in a meaningful way, 
even when they have no words in common. It uses word2vec vector embeddings of words. 
It been shown to outperform many of the state-of-the-art methods in k-nearest neighbors classification .
<br>



### Functions explained:

1.get_best_synset_pair<br>

    Choose the pair with highest path similarity among all pairs.
    Mimics pattern-seeking behavior of humans.
    

2.length_dist<br>

    Return a measure of the length of the shortest path in the semantic
    ontology (Wordnet in our case as well as the paper's) between two
    synsets.
    

3.hierarchy_dist<br>

    Return a measure of depth in the ontology to model the fact that
    nodes closer to the root are broader and have less semantic similarity
    than nodes further away from the root.
    

4.most_similar_word<br>

    Find the word in the joint word set that is most similar to the word
    passed in. We use the algorithm above to compute word similarity between
    the word and each word in the joint word set, and return the most similar
    word and the actual similarity value.

5.info_content<br>

    Uses the Brown corpus available in NLTK to calculate a Laplace
    smoothed frequency distribution of words, then uses this information
    to compute the information content of the lookup_word.
    

6.semantic_vector<br>
    
    Computes the semantic vector of a sentence. The sentence is passed in as
    a collection of words. The size of the semantic vector is the same as the
    size of the joint word set. The elements are 1 if a word in the sentence
    already exists in the joint word set, or the similarity of the word to the
    most similar word in the joint word set if it doesn't. Both values are
    further normalized by the word's (and similar word's) information content
    if info_content_norm is True.
    

7.semantic_similarity<br>

    Computes the semantic similarity between two sentences as the cosine
    similarity between the semantic vectors computed for each sentence.
    

8.word_order_vector<br>
 
    Computes the word order vector for a sentence. The sentence is passed
    in as a collection of words. The size of the word order vector is the
    same as the size of the joint word set. The elements of the word order
    vector are the position mapping (from the windex dictionary) of the
    word in the joint set if the word exists in the sentence. If the word
    does not exist in the sentence, then the value of the element is the
    position of the most similar word in the sentence as long as the similarity
    is above the threshold ETA.
    

9.word_order_similarity<br>
    
    Computes the word-order similarity between two sentences as the normalized
    difference of word order between the two sentences.
    

10.similarity<br>

    Calculate the semantic similarity between two sentences. The last
    parameter is True or False depending on whether information content
    normalization is desired or not.
    


<br>
