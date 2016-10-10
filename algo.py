import nlp
import metrics


class Cluster:
	def __init__(self):
		self.comments = []
		self.terms = {}
		self.center = {}

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
				self.center[term] = 0
			self.terms[term].add(comment)
			self.center[term] += 1

	def remove_comment(self, comment):
		if comment not in self.comments:
			print "Comment does not exist in cluster"
			return
		self.comments.remove(comment)
		for term in self.terms.keys():
			try:
				# print term
				self.terms[term].remove(comment)
				self.center[term] -= 1
			except (ValueError, KeyError):
				pass


	def get_distance_from_center(self,comment):
		comment = nlp.preprocess(comment)
		sim = 0
		term_vector = set(metrics.get_term_vector(comment))
		center_term_vector = set(self.center.keys())
		intersection = term_vector & center_term_vector
		for term in intersection:
			sim = sim + min(self.center[term],2)
		sim = float(sim)
		# print "sim 1", sim
		if sim > metrics.T:
			sim = float(1)
		else:
			sim = sim/float(metrics.T)
		# print "sim 2",sim

		if sim == float(0):
			return float("inf")
		else:
			return (1.0/sim) - 1.0


	def get_radius(self):
		max_dist = float("-inf")
		for comment in self.comments:
			distance = self.get_distance_from_center(comment)
			if distance > max_dist:
				max_dist = distance
		return max_dist

	def get_center(self):
		return self.center

	def print_terms(self):
		print self.terms

	def print_comments(self):
		print self.comments



# set of all the clusters
clusters = set()
radius_threshold = 3.0

def increSTS(new_comment):
	if len(clusters) == 0:
		c = Cluster()
		c.add_comment(new_comment)
		clusters.add(c)
		return
	ca = [cluster for cluster in clusters if cluster.get_distance_from_center(new_comment) != float("inf")]
	cb = [cluster for cluster in ca if cluster.get_distance_from_center(new_comment) < radius_threshold]

	if len(cb) != 0:
		cb.sort(key = lambda c: len(c.comments))
		cadded = cb[0]
		cadded.add_comment(new_comment)
		cchanged = set()
		for cluster in ca:
			if cluster == cadded:
				continue
			for comment in cluster.comments:
				if cadded.get_distance_from_center(comment) < radius_threshold:
					cadded.add_comment(comment)
					cluster.remove_comment(comment)
					cchanged.add(cluster)
		for cluster in cchanged:
			V = [comment for comment in cluster.comments if cluster.get_distance_from_center(comment) > radius_threshold]
			for excluded_comment in V:
				cluster.remove_comment(excluded_comment)
				clusters.sort(key = lambda c: len(c.comments))
				added = False
				for candidate_cluster in clusters:
					if candidate_cluster.get_distance_from_center(excluded_comment) < radius_threshold:
						candidate_cluster.add_comment(excluded_comment)
						added = True
						break
				if not added:
					c = Cluster()
					c.add_comment(new_comment)
					clusters.add(c)
					return

	else:
		c = Cluster()
		c.add_comment(new_comment)
		clusters.add(c)
		return





		





if __name__ == "__main__":
	c = Cluster()
	c.add_comment("This is a cooooomment")
	before = c.terms
	c.add_comment("i love lady gaga")
	c.remove_comment(c.comments[1])
	after = c.terms
