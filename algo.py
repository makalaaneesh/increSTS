import nlp


class Cluster:
	def __init__(self):
		self.comments = []
		self.terms = {}

	def add_comment(self, comment):
		index = len(self.comments)
		comment = nlp.preprocess(comment)
		self.comments.append(comment)

		terms = []
		for i in range(1,4):
			# for 1,2,3
			iterms = nlp.extract_ngrams(comment, i)
			terms  = terms + iterms
		# print terms
		for term in terms:
			cleaned_term = nlp.remove_stopwords(term)
			if len(cleaned_term) == 0:
				continue
			if term not in self.terms.keys():
				self.terms[term] = set()
			self.terms[term].add(index)

	def print_terms(self):
		print self.terms

	def print_comments(self):
		print self.comments



if __name__ == "__main__":
	c = Cluster()
	c.add_comment("This is a comment")
	c.print_terms()