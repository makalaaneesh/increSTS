from nltk import ngrams
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import stem
import re
def extract_ngrams(sentence, n):
	grams = ngrams(sentence.split(), n)
	return list(grams)

def preprocess(comment):
	tokenizer = RegexpTokenizer(r'\w+') # tokenizer that picks out alphanumeric characters as tokens and drop everything else like punctuations
	porter = stem.porter.PorterStemmer()
	comment = comment.lower()
	tokens = tokenizer.tokenize(comment)
	tokens = [replace_three_or_more(token) for token in tokens]
	tokens = [porter.stem(token) for token in tokens]
	comment = " ".join(tokens)
	# print "[[preprocessed = ||", comment, "|| = ]]"
	return comment

def remove_stopwords(term):
	term_list = [word for word in term if word not in stopwords.words('english')]
	return tuple(term_list)


def replace_three_or_more(s):
    # pattern to look for three or more repetitions of any character, including
    # newlines.
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL) 
    return pattern.sub(r"\1", s)





if __name__ == "__main__":

	preprocess("Hello!! THis is a sentence with a lot of punctuations...")
	print type(replace_three_or_more("sooooo"))
	new_tuple = remove_stopwords(('i','am','here'))
	print new_tuple

