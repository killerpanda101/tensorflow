import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
# lemzatizer removes ing,s,etc from the word
import numpy as np 
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000

#creating the lexicon

def create_lexicon(pos,neg):
	lexicon = []
	for fi in [pos,neg]:
		with open(fi,'r') as f:
			contents = f.readlines()
			for l in contents[:hm_lines]:
				all_words = words_tokenize(l.lower())
				lexicon += list(all_words) #this lexicon contains copies of words
	
	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	w_counts = Counter(lexicon)

	for w in W_counts:
		if 1000 > w_counts[w] > 25:
			l2.append(w) #you dont want very common and rare words

	return l2

def sample_handling(saple, lexicon, cassification):
	featureset = []

	with open(sample,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			current_words = word_tokenizer(l.lower())


