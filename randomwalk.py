import nlp
import networkx as nx
from networkx.algorithms import bipartite
import math
import matplotlib.pyplot as plt
from nltk.corpus import gutenberg
import numpy as np
import pdb
# import stringcmp
from collections import OrderedDict
import threading
from datetime import datetime
from multiprocessing.dummy import Pool as ThreadPool
import json
from progressbar import ProgressBar

"""
============================
Random walks implementation
============================

This is a script to implement random walks

"""


NGRAM = 5
word_list = {}
THRESHOLD_VALUE = 10
GRAPH_SETTING=False
STEPS_VALUE = 5
MAX_HITS = 4
MAX_WORDS = 4
MAX_THREAD_POOL = 60

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
	f = open("mixedcorpus.txt", "r")
	# f = open("commentstest.txt", "r")
	x = f.read()
	ngrams_list=[]
	for line in x.split("\n"):
		# if ":" not in line:
		# 	continue
		# line = line[line.index(":")+1:]
		line = line.strip()
		line = line.decode("utf-8")
		line = line.encode('ascii','ignore')
		line = nlp.clean(line)
		ngrams_list.append(nlp.extract_ngrams(line,NGRAM))
	return ngrams_list

def isNoisyWord():
	"""
	Noisy word detection
	Check if the word is noisy
	Arguments:
		word string -- [A variable]
	"""
	emma = gutenberg.words('austen-emma.txt')
	watson_lines = open("cleancorpus.txt","r").read().split("\n")
	for watson in watson_lines:
		watson = watson.split(" ")
		for word in watson:
			if word in word_list:
				word_list[word]=word_list[word] + 1
			else:
				word_list[word]=1

	flag=0
	for word in emma:
		if word in word_list:
			word_list[word]=word_list[word] + 1
		else:
			word_list[word]=1
	print "Done creating word list"


def constructGraph(list_of_ngrams,ngram_size):
	"""
	Construct a Graph
	Construct graph from the ngram list
	Arguments:
		list_of_ngrams {list} -- List of ngrams
		ngram_size {int} -- ngram size
	"""
	# isNoisyWord()
	# print word_list
	B = nx.Graph()
	ContextNode_list={}
	WordNode_list={}
	reap_node=0
	reap_word=0
	center_index = int(math.floor(ngram_size/2))
	total_ngrams = len(list_of_ngrams)
	print "No. of n grams "+ str(total_ngrams)
	bar = ProgressBar()
	for n_set in bar(list_of_ngrams):
		if len(n_set)>0:
			for n in n_set:
				# print 
				# print 
				# print "**************************************"
				# print n
				reap_node=0
				reap_word=0
				n=list(n)
				word = n[center_index]
				n[center_index]='*'
				c=None
				if str(n) in ContextNode_list:
					# print "Repeated node"
					reap_node=1
					c=ContextNode_list[str(n)]
				else:
					c = ContextNode(n)
					B.add_node(c,bipartite=0)
					ContextNode_list[str(n)]=c
				w = None
				if word in WordNode_list:
					# print "Repeated word"
					reap_word=1
					w = WordNode_list[word]
				else:
					w = WordNode(word,not(word in word_list and word_list[word]>THRESHOLD_VALUE))
					B.add_node(w,bipartite=1)
					WordNode_list[word]=w
				#Weight 
				if reap_node==1 and reap_word==1 and (w in B[c]):
					# print "Testing"
					B[c][w]['weight'] += 1
				else:
					B.add_edge(c, w, weight=1)
				# bottom_nodes, top_nodes = bipartite.sets(B)
				# print bottom_nodes
				# print top_nodes
	#Debug statemtns
	print "Done****************"
	# X = set(n for n,d in B.nodes(data=True) if d['bipartite']==0)
	# Y = set(B) - X
	# print bipartite.is_bipartite(B)
	# pos = dict()
	# print X
	# print "-----------------------------------"
	# print Y
	# print "-----------------------------------"
	# print B.edges()
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
	hit_matrix = np.array([[0. for node1 in node_list] for node2 in node_list])
	r_matrix = np.array([[0. for node1 in node_list] for node2 in node_list])
	norm_matrix = np.array([[0. for node1 in node_list] for node2 in node_list])
	cost_matrix = np.array([[0. for node1 in node_list] for node2 in node_list])
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
	word_node_list = []

	#Getting only the word node indices
	for i in range(0,len(node_list)):
		if type(node_list[i]) is WordNode:
			word_node_list.append(i)

	#Building the cost matrix
	for i in word_node_list:
		for j in word_node_list:
			if i != j:
				n = str(node_list[i])
				m = str(node_list[j])
				lcs_n_m = len(nlp.lcs(n,m))
				max_length = max(len(n),len(m))
				lcsr_n_m = float(lcs_n_m)/float(max_length)
				edit_n_m = nlp.editex("".join(OrderedDict.fromkeys(n)),"".join(OrderedDict.fromkeys(m)))
				if edit_n_m !=0:
					sim_cost_n_m = lcsr_n_m/edit_n_m
					print sim_cost_n_m
					cost_matrix[i,j] = float(H_matrix[i,j] + sim_cost_n_m)
					# print "="
					# print cost_matrix[i,j]
	print "===========Final Cost Matrix================"
	print cost_matrix
	#Cost matrix done
	for node_index in range(0,len(node_list)):
		if type(node_list[node_index]) is WordNode and node_list[node_index].isNoisy:
			final_word_map[str(node_list[node_index])]=[]
			# pdb.set_trace()
			row_array = cost_matrix[node_index,None,:]
			row_array = np.asarray(np.argsort(row_array,axis=1)).reshape(-1)[::-1]
			for word_index in range(0,MAX_WORDS):
				if type(node_list[row_array[word_index]]) is WordNode and not node_list[row_array[word_index]].isNoisy:
					final_word_map[str(node_list[node_index])].append((str(node_list[row_array[word_index]]),cost_matrix[node_index,row_array[word_index]]))
	print final_word_map


def compute_probabilities(row):
	total = row.sum()
	newrow = np.true_divide(row, total)
	np.put(row, range(0,len(row)), newrow)


class MatrixPower(threading.Thread):
	"""docstring for Client"""
	def __init__(self,A, P, i, index):
		threading.Thread.__init__(self)
		# self.private_key = private_key
		self.A = A
		self.P = P
		self.i = i
		self.index = index
		# print "Thread id : "+ str(self.thread_count)

	def run(self):
		new_matrix = np.linalg.matrix_power(self.A,self.i)
		self.P[self.index] = new_matrix 


def init_randomwalk(B):
	print "INITING RANDOM WALK"
	A = nx.to_numpy_matrix(B)
	# print A
	node_list = B.nodes()
	node_len = len(node_list)

	pool = ThreadPool(MAX_THREAD_POOL)
	rows = [row for row in A]
	pool.map(compute_probabilities, rows)
	pool.close()
	pool.join()
	# for i in range(0,len(node_list)):
	# 	total = 0.0
	# 	total = A[i,:].sum()
	# 	# for j in range(0,len(node_list)):
	# 	# 	total = total + A[i,j]
	# 	if total!=0:
	# 		A[i,:] = np.true_divide(A[i,:], total)
			# for j in range(0,len(node_list)):
			# 	A[i,j] = A[i,j]/total
	# print A
	#Random walks algorithm
	print "COMPLTED THREADS TO COMPUTE PROBABILITIES"
	A_mask = np.ma.masked_where(A==0., A)
	
	powers = range(1,STEPS_VALUE+1,2)
	P = [None for i in powers]
	mxthreads = []
	for index, i in enumerate(powers):
		mx_thread = MatrixPower(A, P , i, index)
		mxthreads.append(mx_thread)
		mx_thread.start()
		mx_thread.join()

		
	for thread in mxthreads:
		thread.join()

	for p in P:
		p = np.ma.masked_array(p, A_mask.mask)
		np.ma.set_fill_value(p, 0.)
		p = p.filled()
	print "COMPLETED INITING RANDOM WALK"
	return A,A_mask,node_list,P


def execute_thread(thread):
	thread.start()
	thread.join()

def randomwalk_thread(B,node_arr,start_time,graph_time,init_time):
	"""Random walk threading function
	
	Threaded approach to random walk to improve speed
	
	Arguments:
		B {Graph} -- Starting graph
	"""
	#Create weight probabities:
	A,A_mask,node_list,P_arr = init_randomwalk(B)
	random_walk_threads = []
	final_word_map = {}
	total_count = 0
	# for node_index in node_arr:
	# 	if type(node_list[node_index]) is WordNode and node_list[node_index].isNoisy:
	# 		total_count = total_count + 1
	# print "Max threads created " + str(total_count)
	for i, node_index in enumerate(node_arr):
		if type(node_list[node_index]) is WordNode and node_list[node_index].isNoisy:
			r = RandomWalk(node_index,A,A_mask,node_list,P_arr,final_word_map,start_time)
			print "Created {i}th thread".format(i = i)
			# r.start()
			random_walk_threads.append(r)

	print "STARTING THREADSXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
	p = ThreadPool(MAX_THREAD_POOL)
	p.map(execute_thread, random_walk_threads)
	p.close()
	p.join()

	# for i in range(0,len(random_walk_threads),MAX_THREAD_POOL):
	# 	for j in range(i,i+MAX_THREAD_POOL):
	# 		random_walk_threads[j].start()
	# 	for j in range(i,i+MAX_THREAD_POOL):
	# 		random_walk_threads[j].join()
	# 	print "One pool of threads completed"
	# print "**********Total threads : "+ str(len(random_walk_threads))
	# for r in random_walk_threads:
	# 	r.join()
	print "FINAL WORD MAPXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
	dict_str = json.dumps(final_word_map)
	f = open("finaldictionary.json", "w")
	f.write(dict_str)
	f.close()
	print final_word_map
	end_time = datetime.now()
	time_delta = end_time - start_time
	time_delta_2 = start_time - graph_time
	time_delta_3 = end_time - init_time
	print "Total randomwalk time taken: "+ str(time_delta.seconds)+"s"
	print "Total graph time taken: "+ str(time_delta_2.seconds)+"s"
	print "Total time taken: "+ str(time_delta_3.seconds)+"s"


class RandomWalk(threading.Thread):
	"""docstring for Client"""
	def __init__(self,node_index,A,A_mask,node_list,P_arr,final_word_map,start_time):
		threading.Thread.__init__(self)
		# self.private_key = private_key
		self.node_index = node_index
		self.A = A
		self.A_mask = A_mask
		self.node_list = node_list
		self.P_arr = P_arr
		self.final_word_map = final_word_map
		self.start_time = start_time
		# print "Thread id : "+ str(self.thread_count)

	def run(self):
		print "Created Random walk thread"
		node_len = len(self.node_list)
		hit_matrix = np.zeros(node_len)
		r_matrix = np.zeros(node_len)
		norm_matrix = np.zeros(node_len)
		cost_matrix = np.zeros(node_len)
		node_index = self.node_index
		start_node_index = node_index
		source_node_index = node_index
		for i in range(0,(STEPS_VALUE/2)+1):
			start_node_index = node_index
			source_node_index = node_index
			P = self.P_arr[i]
			# P = np.ma.masked_array(P, self.A_mask.mask)
			# np.ma.set_fill_value(P, 0.)
			# P = P.filled()
			hits = 0
			print "STEP "+str(i)
			print self.node_list[start_node_index]
			while (type(self.node_list[source_node_index]) is ContextNode) or (type(self.node_list[source_node_index]) is WordNode and self.node_list[source_node_index].isNoisy) or (hits < MAX_HITS):
				hits = hits + 1
				row_array = P[source_node_index,None,:]
				row_array[0,start_node_index]=0
				source_node_index = np.argmax(row_array)
				if row_array[0,source_node_index] == 0:
					print "No where to go"
					break
				print "->"
				print self.node_list[source_node_index]
				# pdb.set_trace()
				if (type(self.node_list[source_node_index]) is WordNode and not self.node_list[source_node_index].isNoisy) or (hits >= MAX_HITS):
					break
			print "STEP Done"
			r_matrix[source_node_index]=r_matrix[source_node_index]+1
			hit_matrix[source_node_index]=hits
		H_matrix = np.true_divide(hit_matrix,r_matrix)
		where_are_NaNs = np.isnan(H_matrix)
		H_matrix[where_are_NaNs] = 0.
		print "==========Final H Matrix==========="
		print H_matrix
		total = 0.0
		for j in range(0,len(self.node_list)):
			total = total + H_matrix[j]
		if total!=0:
			for j in range(0,len(self.node_list)):
				H_matrix[j] = H_matrix[j]/total
		print H_matrix
		final_word_map={}
		word_node_list = []

		#Getting only the word node indices
		for i in range(0,len(self.node_list)):
			if type(self.node_list[i]) is WordNode:
				word_node_list.append(i)

		#Building the cost matrix
		for j in word_node_list:
			if node_index != j:
				n = str(self.node_list[node_index])
				m = str(self.node_list[j])
				lcs_n_m = len(nlp.lcs(n,m))
				max_length = max(len(n),len(m))
				lcsr_n_m = float(lcs_n_m)/float(max_length)
				edit_n_m = nlp.editex("".join(OrderedDict.fromkeys(n)),"".join(OrderedDict.fromkeys(m)))
				if edit_n_m !=0:
					sim_cost_n_m = lcsr_n_m/edit_n_m
					print sim_cost_n_m
					cost_matrix[j] = float(H_matrix[j] + sim_cost_n_m)
					# print "="
					# print cost_matrix[i,j]
		print "===========Final Cost Matrix================"
		print cost_matrix
		#Cost matrix done
		if type(self.node_list[node_index]) is WordNode and self.node_list[node_index].isNoisy:
			self.final_word_map[str(self.node_list[node_index])]=[]
			# pdb.set_trace()
			row_array = cost_matrix
			row_array = np.asarray(np.argsort(row_array)).reshape(-1)[::-1]
			for word_index in range(0,MAX_WORDS):
				if type(self.node_list[row_array[word_index]]) is WordNode and not self.node_list[row_array[word_index]].isNoisy:
					self.final_word_map[str(self.node_list[node_index])].append((str(self.node_list[row_array[word_index]]),cost_matrix[row_array[word_index]]))
		print self.final_word_map
		end_time = datetime.now()
		time_delta = end_time - self.start_time
		print "Total time taken: "+ str(time_delta.seconds)+"s"



if __name__ == "__main__":
	#Step 1: Get all ngrams from the text corpus
	start_time_start = datetime.now()
	print "Starting time : "+ str(start_time_start)
	list_of_ngrams=getallngrams()
	isNoisyWord()
	# print list_of_ngrams
	graph_time = datetime.now()
	print "Graph Starting time : "+ str(graph_time)
	B,X,Y=constructGraph(list_of_ngrams,NGRAM)
	indexes = [i for i in range(0,len(B.nodes()))]
	# randomwalk(B,X,Y)
	start_time = datetime.now()
	print "Random walk Starting time : "+ str(start_time)
	randomwalk_thread(B,indexes,start_time,graph_time,start_time_start)


