import nlp

# constants
D = 2
T = 1
NGRAM = 5

def get_term_vector(comment):
	# print "COMMENT", "[[[", comment, "]]]"
	terms = []
	for i in range(1,NGRAM+1):
		# for 1,2,3
		iterms = nlp.extract_ngrams(comment, i)
		terms  = terms + iterms
	# print terms
	terms = [term for term in terms if len(nlp.remove_stopwords(term)) > 0]
	stemmed_terms = []
	for term in terms:
		stemmed_term = tuple(map(nlp.porter_stem, term))
		stemmed_terms.append(stemmed_term)
	return stemmed_terms


def comment_comment_similarity(comment1, comment2):
	comment1 = nlp.preprocess(comment1)
	comment2 = nlp.preprocess(comment2)
	term_vector1 = set(get_term_vector(comment1))
	term_vector2 = set(get_term_vector(comment2))
	intersection = term_vector1 & term_vector2
	sim = float(len(intersection))
	if sim > D:
		val = 1
	else:
		val = sim/float(D)
	return val


def comment_comment_distance(comment1, commment2):
	sim = comment_comment_similarity(comment1, comment2)
	if sim == float(0):
		return float("inf")
	else:
		return (1.0/sim) - 1.0





print get_term_vector("THis is a comment")
comment_comment_similarity("This is a comment", "Hey I am a comment")