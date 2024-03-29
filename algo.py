import nlp
import metrics
import sentiment

class Cluster:
	def __init__(self):
		self.comments = []
		self.comment_term_vectors = []
		self.terms = {}
		self.center = {}
		self.sentiment_score = 0.0
		self.emoticonlist = {}
		with open('emoticonlist') as f:
			lines = f.readlines()
			for l in lines:
				row = l.split(">")
				self.emoticonlist[row[0]] = row[1]

	def add_comment(self, comment):
		index = len(self.comments)
		complete_sentiment = self.sentiment_score* index
		emoticon_sentiment = sentiment.get_emoticon_score(comment,self.emoticonlist)
		if emoticon_sentiment > 0:
			complete_sentiment = complete_sentiment + (emoticon_sentiment + sentiment.get_sentiment_score(comment))/2.0
		else:
			complete_sentiment = complete_sentiment + sentiment.get_sentiment_score(comment)
		self.sentiment_score = float(complete_sentiment)/float(index+1)
		# comment = nlp.preprocess(comment)
		self.comments.append(comment)

		term_vector = list(set(metrics.get_term_vector(comment)))
		self.comment_term_vectors.append(term_vector)

		# terms = []
		# for i in range(1,4):
		# 	# for 1,2,3
		# 	iterms = nlp.extract_ngrams(comment, i)
		# 	terms  = terms + iterms
		# print terms
		for term in term_vector:
			# cleaned_term = nlp.remove_stopwords(term)
			# if len(cleaned_term) == 0:
			# 	continue
			if term not in self.terms.keys():
				self.terms[term] = set()
				self.center[term] = 0
			self.terms[term].add(comment)
			self.center[term] += 1

	def remove_comment(self, comment, index):
		if comment not in self.comments:
			print "Comment does not exist in cluster"
			return

		# index = self.comments.index(comment)
		# self.comments.remove(comment)
		emoticon_sentiment = sentiment.get_emoticon_score(comment,self.emoticonlist)
		complete_sentiment = self.sentiment_score* index
		if emoticon_sentiment > 0:
			complete_sentiment = complete_sentiment - (emoticon_sentiment + sentiment.get_sentiment_score(comment))/2.0
		else:
			complete_sentiment = complete_sentiment - sentiment.get_sentiment_score(comment)
		try:
			self.sentiment_score = float(complete_sentiment)/float(index-1)
		except:
			self.sentiment_score = 0

		del self.comments[index]
		tv =  self.comment_term_vectors[index]
		# print tv
		# print list(clusters).index(self)
		

		for term in tv:
			try:
				# print term
				self.terms[term].remove(comment)
				self.center[term] -= 1
			except (ValueError, KeyError):
				pass

		del self.comment_term_vectors[index]



	# compute term vector of comment before hand and pass it to the function as this will be called for each cluster
	def get_distance_from_center(self,comment, term_vector=None):
		# comment = nlp.preprocess(comment)
		sim = 0
		if term_vector is None:
			term_vector = set(metrics.get_term_vector(comment))
		center_term_vector = set(self.center.keys())
		intersection = term_vector & center_term_vector
		common_terms = len(intersection)
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
			return float("inf"),common_terms
		else:
			return (1.0/sim) - 1.0, common_terms


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


def get_comments():
	comments = []
	f = open("ultracleanedcomments.txt", "r")
	x = f.read()
	for line in x.split("\n"):
		# if ":" not in line:
		# 	print "ignored"
		# 	continue
		# line = line[line.index(":")+1:]
		line = line.strip()
		line = line.decode("utf-8")
		line = line.encode("ascii","ignore")
		comments.append(line)
	# print comments
	return comments

# radius_threshold = 3.0
TH_TERMS  = 7

def increSTS(new_comment, clusters):
	new_comment = nlp.preprocess(new_comment)
	term_vector = set(metrics.get_term_vector(new_comment))
	if len(clusters) == 0:
		c = Cluster()
		c.add_comment(new_comment)
		clusters.add(c)
		return

	ca = []
	cb = []
	for cluster in clusters:
		dist, terms = cluster.get_distance_from_center(new_comment, term_vector)
		if dist != float("inf"):
			ca.append(cluster)
		# if dist < radius_threshold:
		if terms > TH_TERMS:
			cb.append(cluster)
	# ca = [cluster for cluster in clusters if cluster.get_distance_from_center(new_comment, term_vector) != float("inf")]
	# cb = [cluster for cluster in ca if cluster.get_distance_from_center(new_comment,term_vector) < radius_threshold]

	if len(cb) != 0:
		cb.sort(key = lambda c: len(c.comments),reverse = True)
		cadded = cb[0]
		cadded.add_comment(new_comment)
		cchanged = set()
		for cluster in ca:
			if cluster == cadded:
				continue
			for i, comment in enumerate(cluster.comments):
				tv = set(cluster.comment_term_vectors[i])
				# if cadded.get_distance_from_center(comment,tv)[0] < radius_threshold:
				if cadded.get_distance_from_center(comment,tv)[1] > TH_TERMS:
					cadded.add_comment(comment)
					cluster.remove_comment(comment,i)
					cchanged.add(cluster)
		for cluster in cchanged:
			# V = [comment for comment in cluster.comments if cluster.get_distance_from_center(comment) > radius_threshold]
			V = []
			Vindex = []
			Vtv = []
			for i, comment in enumerate(cluster.comments):
				tv = set(cluster.comment_term_vectors[i])
				# if cluster.get_distance_from_center(comment, tv)[0] >= radius_threshold:
				if cluster.get_distance_from_center(comment, tv)[1] <= TH_TERMS:
					V.append(comment)
					Vindex.append(i)
					Vtv.append(tv)
			while len(V) > 0:
				for i, excluded_comment in enumerate(V):
					excluded_comment_tv = Vtv[i]
					cluster.remove_comment(excluded_comment, Vindex[i])
					clusters_list = list(clusters)
					clusters_list.sort(key = lambda c: len(c.comments),reverse = True)
					added = False
					for candidate_cluster in clusters_list:
						# if candidate_cluster.get_distance_from_center(excluded_comment,excluded_comment_tv)[0] < radius_threshold:
						if candidate_cluster.get_distance_from_center(excluded_comment,excluded_comment_tv)[1] > TH_TERMS:
							candidate_cluster.add_comment(excluded_comment)
							added = True
							break
					if not added:
						c = Cluster()
						c.add_comment(new_comment)
						clusters.add(c)
						
				V = []
				Vtv = []
				for i, comment in enumerate(cluster.comments):
					tv = set(cluster.comment_term_vectors[i])
					# if cluster.get_distance_from_center(comment, tv)[0] >= radius_threshold:
					if cluster.get_distance_from_center(comment, tv)[1] <= TH_TERMS:
						V.append(comment)
						Vtv.append(tv)

	else:
		c = Cluster()
		c.add_comment(new_comment)
		clusters.add(c)
		return





		





if __name__ == "__main__":
	# c = Cluster()
	# c.add_comment("This is a cooooomment")
	# before = c.terms
	# c.add_comment("i love lady gaga")
	# c.remove_comment(c.comments[1])
	# after = c.terms


	# set of all the clusters
	clusters = set()
	comments = get_comments()
	# of = open("cleanedcomments.txt", "w")

	for i, comment in enumerate(comments):
		# of.write(comment+"\n")
		increSTS(comment, clusters)
		print i," iteration complete---------", 
		print "{len} clusters created".format(len=len(clusters))

	cl = list(clusters)
	cl.sort(key = lambda c: len(c.comments),reverse = True)
	f = open("clusters.txt","w")
	for i in range(0,16):
		f.write("\n-------------"+str(i)+"---------------\n")
		f.write("\n".join(cl[i].comments))
		f.write("\n--------------[{"+str(cl[i].sentiment_score)+"}]------------------\n")

	f.close()





#radius threshold 2.5 and T = 50 worked better with Rihanna
#radius threshold 3.0 and T = 40 gave good results for Disney





# [Tue Oct 18 15:09:30.603955 2016] [:error] [pid 10966] {'status': 200, 'transaction_amount': 1250L, 'hash': 'ba2c8dcc7b1554f5cc08c0c9f3a330821b76b606c2a64a36a3b9ee8c850f4c533110cbd7b64a53e853c8d67dab4a4147e7987e30856c125d9913de07682d493b', 'message': 'Transaction created', 'salt': 'vlqHgOcF', 'transaction_id': 'a29a9fa04d2a4a5098ba2a5280b7a'}