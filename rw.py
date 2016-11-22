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
from graphclass import ContextNode,WordNode
import time

NGRAM = 5
word_list = {}
THRESHOLD_VALUE = 10
GRAPH_SETTING=False
STEPS_VALUE = 5
MAX_HITS = 4
MAX_WORDS = 4
MAX_THREAD_POOL = 50

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
		# return
		import nlp
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
			P = np.ma.masked_array(P, self.A_mask.mask)
			np.ma.set_fill_value(P, 0.)
			P = P.filled()
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