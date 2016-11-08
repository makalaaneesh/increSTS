from nltk import ngrams
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import stem 
import re
import string


def lcs(a, b):
    lengths = [[0 for j in range(len(b)+1)] for i in range(len(a)+1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
    # read the substring out from the matrix
    result = ""
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x-1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y-1]:
            y -= 1
        else:
            assert a[x-1] == b[y-1]
            result = a[x-1] + result
            x -= 1
            y -= 1
    return result



regex = re.compile('[%s]' % re.escape(string.punctuation)) 


def extract_ngrams(sentence, n):
	grams = ngrams(sentence.split(), n)
	return list(grams)


def porter_stem(word):
	porter = stem.porter.PorterStemmer()
	return porter.stem(word)

def preprocess(comment):
	# tokenizer = RegexpTokenizer(r'\w+') # tokenizer that picks out alphanumeric characters as tokens and drop everything else like punctuations
	
	comment = comment.lower()

	comment = regex.sub('',comment)
	tokens = word_tokenize(comment)
	# tokens = tokenizer.tokenize(comment)
	tokens = [replace_three_or_more(token) for token in tokens]
	# tokens = [porter.stem(token) for token in tokens]
	comment = " ".join(tokens)
	comment = comment.strip()
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

	# preprocess("Hello!! THis is a sentence with a lot of punctuations...")
	# print type(replace_three_or_more("sooooo"))
	# new_tuple = remove_stopwords(('i','am','here'))
	# print new_tuple

	print lcs("AGGTAB","GXTXAYB")

