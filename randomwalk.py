import nlp
import networkx as nx
from networkx.algorithms import bipartite
import math
import matplotlib.pyplot as plt
from nltk.corpus import gutenberg
import numpy as np
import pdb

"""
============================
Random walks implementation
============================

This is a script to implement random walks

"""



word_list = {}
THRESHOLD_VALUE = 10
GRAPH_SETTING=False
STEPS_VALUE = 5
MAX_HITS = 4
MAX_WORDS = 4

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
		return hash(str(self.context_list))
	def __repr__(self):
		return str(self.context_list)
	def __str__(self):
		return str(self.context_list)

	def displayNode():
		print context_list

class WordNode(object):
	"""docstring for ContextNode"""
	word=""
	isNoisy=True
	def __init__(self, word,isNoisy):
		super(WordNode, self).__init__()
		self.word = word
		self.isNoisy = isNoisy
	def __eq__(self, other):
		"""Override the default Equals behavior"""
		if isinstance(other, self.__class__):
		    return str(self.word) == str(other.word)
		return NotImplemented

	def __ne__(self, other):
		"""Define a non-equality test"""
		if isinstance(other, self.__class__):
		    return not self.__eq__(other)
		return NotImplemented

	def __hash__(self):
		"""Override the default hash behavior (that returns the id or the object)"""
		return hash(str(self.word))
	def __repr__(self):
		return self.word
	def __str__(self):
		return self.word

	def displayNode():
		print word
		


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

def isNoisyWord():
	"""
	Noisy word detection
	Check if the word is noisy
	Arguments:
		word string -- [A variable]
	"""
	emma = gutenberg.words('austen-emma.txt')
	flag=0
	for word in emma:
		if word in word_list.keys():
			word_list[word]=word_list[word] + 1
		else:
			word_list[word]=0


def constructGraph(list_of_ngrams,ngram_size):
	"""
	Construct a Graph
	Construct graph from the ngram list
	Arguments:
		list_of_ngrams {list} -- List of ngrams
		ngram_size {int} -- ngram size
	"""
	isNoisyWord()
	# print word_list
	B = nx.Graph()
	ContextNode_list={}
	WordNode_list={}
	reap_node=0
	reap_word=0
	center_index = int(math.floor(ngram_size/2))
	for n_set in list_of_ngrams:
		if len(n_set)>0:
			for n in n_set:
				print 
				print 
				print "**************************************"
				print n
				reap_node=0
				reap_word=0
				n=list(n)
				word = n[center_index]
				n[center_index]='*'
				c=None
				if str(n) in ContextNode_list:
					print "Repeated node"
					reap_node=1
					c=ContextNode_list[str(n)]
				else:
					c = ContextNode(n)
					B.add_node(c,bipartite=0)
					ContextNode_list[str(n)]=c
				w = None
				if word in WordNode_list:
					print "Repeated word"
					reap_word=1
					w = WordNode_list[word]
				else:
					w = WordNode(word,not(word in word_list and word_list[word]>THRESHOLD_VALUE))
					B.add_node(w,bipartite=1)
					WordNode_list[word]=w
				#Weight 
				if reap_node==1 and reap_word==1 and (w in B[c]):
					print "Testing"
					B[c][w]['weight'] += 1
				else:
					B.add_edge(c, w, weight=1)
				bottom_nodes, top_nodes = bipartite.sets(B)
				# print bottom_nodes
				# print top_nodes
	#Debug statemtns
	X = set(n for n,d in B.nodes(data=True) if d['bipartite']==0)
	Y = set(B) - X
	print bipartite.is_bipartite(B)
	# pos = dict()
	print X
	print "-----------------------------------"
	print Y
	print "-----------------------------------"
	print B.edges()
	if GRAPH_SETTING==True:
		val_map = {'textnoisy': 1.0,
	           'context': 0.5714285714285714,
	           'textabsorbing': 0.0,}
		values = []
		for node in B.nodes():
			if isinstance(node,ContextNode):
				values.append(val_map['context'])
				continue
			if isinstance(node,WordNode):
				if node.isNoisy:
					values.append(val_map['textnoisy'])
				else:
					values.append(val_map['textabsorbing'])
		#Draw the graph
		pos=nx.spring_layout(B)
		nx.draw_networkx_nodes(B,pos,node_size=700,node_color=values)
		edge_labels=dict([((u,v,),d['weight'])
	             for u,v,d in B.edges(data=True)])
		nx.draw_networkx_edge_labels(B,pos,edge_labels=edge_labels)
		nx.draw_networkx_edges(B,pos)
		nx.draw_networkx_labels(B,pos,font_size=20,font_family='sans-serif')
		plt.axis('off')
		plt.show()
	return B,ContextNode_list,WordNode_list


def randomwalk(B,X,Y):
	"""Random walk implementation
	
	This is the random walk implementation
	Arguments:
		B {Networkx graphs} -- Networkx graph that is created
		X {WordNodes} -- List of WordNode
		Y {ContextNodes} -- List of ContextNodes
	"""
	#Create weight probabities:
	A = nx.to_numpy_matrix(B)
	print A
	node_list = B.nodes()
	hit_matrix = np.array([[0 for node1 in node_list] for node2 in node_list])
	r_matrix = np.array([[0 for node1 in node_list] for node2 in node_list])
	norm_matrix = np.array([[0 for node1 in node_list] for node2 in node_list])
	cost_matrix = np.array([[0 for node1 in node_list] for node2 in node_list])
	for i in range(0,len(node_list)):
		total = 0.0
		for j in range(0,len(node_list)):
			total = total + A[i,j]
		if total!=0:
			for j in range(0,len(node_list)):
				A[i,j] = A[i,j]/total
	print A
	#Random walks algorithm
	A_mask = np.ma.masked_where(A==0., A)

	for node_index in range(0,len(node_list)):
		if type(node_list[node_index]) is WordNode and node_list[node_index].isNoisy:
			start_node_index = node_index
			source_node_index = node_index
			for i in range(1,STEPS_VALUE+1,2):
				start_node_index = node_index
				source_node_index = node_index
				P = np.linalg.matrix_power(A,i)
				P = np.ma.masked_array(P, A_mask.mask)
				np.ma.set_fill_value(P, 0.)
				P = P.filled()
				hits = 0
				print "STEP "+str(i)
				print node_list[start_node_index]
				while (type(node_list[source_node_index]) is ContextNode) or (type(node_list[source_node_index]) is WordNode and node_list[source_node_index].isNoisy) or (hits < MAX_HITS):
					hits = hits + 1
					row_array = P[source_node_index,None,:]
					row_array[0,start_node_index]=0
					source_node_index = np.argmax(row_array)
					if row_array[0,source_node_index] == 0:
						print "No where to go"
						break
					print "->"
					print node_list[source_node_index]
					# pdb.set_trace()
					if (type(node_list[source_node_index]) is WordNode and not node_list[source_node_index].isNoisy) or (hits >= MAX_HITS):
						break
				print "STEP Done"
				r_matrix[start_node_index,source_node_index]=r_matrix[start_node_index,source_node_index]+1
				hit_matrix[start_node_index,source_node_index]=hits
	H_matrix = np.true_divide(hit_matrix,r_matrix)
	where_are_NaNs = np.isnan(H_matrix)
	H_matrix[where_are_NaNs] = 0.
	print "==========Final H Matrix==========="
	print H_matrix
	for i in range(0,len(node_list)):
		total = 0.0
		for j in range(0,len(node_list)):
			total = total + H_matrix[i,j]
		if total!=0:
			for j in range(0,len(node_list)):
				H_matrix[i,j] = H_matrix[i,j]/total
	print H_matrix
	final_word_map={}
	for node_index in range(0,len(node_list)):
		if type(node_list[node_index]) is WordNode and node_list[node_index].isNoisy:
			final_word_map[str(node_list[node_index])]=[]
			# pdb.set_trace()
			row_array = H_matrix[node_index,None,:]
			row_array = np.asarray(np.argsort(row_array,axis=1)).reshape(-1)[::-1]
			for word_index in range(0,MAX_WORDS):
				final_word_map[str(node_list[node_index])].append(str(node_list[row_array[word_index]]))
	print final_word_map



if __name__ == "__main__":
	#Step 1: Get all ngrams from the text corpus
	list_of_ngrams=getallngrams()
	print list_of_ngrams
	B,X,Y=constructGraph(list_of_ngrams,3)
	randomwalk(B,X,Y)


