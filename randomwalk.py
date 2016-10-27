import nlp
import networkx as nx
from networkx.algorithms import bipartite
import math
import matplotlib.pyplot as plt

"""
============================
Random walks implementation
============================

This is a script to implement random walks

"""



word_list = {}
THRESHOLD_VALUE = 10

class ContextNode(object):
	"""docstring for ContextNode"""
	context_list=[]
	def __init__(self, context_list):
		super(ContextNode, self).__init__()
		self.context_list = context_list
	def __eq__(self, other):
		"""Override the default Equals behavior"""
		if isinstance(other, self.__class__):
		    return set(self.context_list) == set(other.context_list)
		return NotImplemented

	def __ne__(self, other):
		"""Define a non-equality test"""
		if isinstance(other, self.__class__):
		    return not self.__eq__(other)
		return NotImplemented

	def __hash__(self):
		"""Override the default hash behavior (that returns the id or the object)"""
		return hash(tuple(sorted(self.context_list)))
	def __repr__(self):
		return str(self.context_list)
	def __str__(self):
		return str(self.context_list)

	def displayNode():
		print context_list
		


def getallngrams():
	"""
	Getting all the ngrams
	"""
	f = open("commentstest.txt", "r")
	x = f.read()
	ngrams_list=[]
	for line in x.split("\n"):
		# if ":" not in line:
		# 	continue
		# line = line[line.index(":")+1:]
		line = line.strip()
		line = line.decode("utf-8")
		ngrams_list.append(nlp.extract_ngrams(line,3))
	return ngrams_list

def isNoisyWord(word):
	"""
	Noisy word detection
	Check if the word is noisy
	Arguments:
		word string -- [A variable]
	"""
	flag=0
	if word in word_list.keys():
		if word_list[word] > THRESHOLD_VALUE:
			flag=1
		word_list[word]=word_list[word] + 1
	else:
		word_list[word]=0


def constructGraph(list_of_ngrams,ngram_size):
	for n in list_of_ngrams:
		print n
		for word in n:
			for w in word:
				isNoisyWord(w)
	print word_list
	B = nx.Graph()
	center_index = int(math.floor(ngram_size/2))
	for n_set in list_of_ngrams:
		if len(n_set)>0:
			for n in n_set:
				print n
				n=list(n)
				word = n[center_index]
				n[center_index]='*'
				c = ContextNode(n)
				B.add_node(c,bipartite=0)
				B.add_node(word,bipartite=1)
				if B.has_edge(c, word):
					B[c][word]['weight'] += 1
				else:
					B.add_edge(c, word, weight=1)
				bottom_nodes, top_nodes = bipartite.sets(B)
				# print bottom_nodes
				# print top_nodes
	X = set(n for n,d in B.nodes(data=True) if d['bipartite']==0)
	Y = set(B) - X
	print bipartite.is_bipartite(B)
	# pos = dict()
	print X
	print "-----------------------------------"
	print Y
	print "-----------------------------------"
	print B.edges()
	# pos.update( (n, (1, i)) for i, n in enumerate(X) ) # put nodes from X at x=1
	# pos.update( (n, (2, i)) for i, n in enumerate(Y) ) # put nodes from Y at x=2
	# nx.draw(B, pos=pos)
	# plt.show()
	pos=nx.spring_layout(B) # positions for all nodes

	# nodes
	nx.draw_networkx_nodes(B,pos,node_size=700)

	# edges
	# nx.draw_networkx_edges(B,pos,edgelist=B.edges(),width=6)
	edge_labels=dict([((u,v,),d['weight'])
             for u,v,d in B.edges(data=True)])
	nx.draw_networkx_edge_labels(B,pos,edge_labels=edge_labels)
	# labels
	nx.draw_networkx_labels(B,pos,font_size=20,font_family='sans-serif')

	plt.axis('off')
	plt.show()

if __name__ == "__main__":
	#Step 1: Get all ngrams from the text corpus
	list_of_ngrams=getallngrams()
	print list_of_ngrams
	constructGraph(list_of_ngrams,3)


