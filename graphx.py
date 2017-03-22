"""
spark-submit --packages graphframes:graphframes:0.2.0-spark2.0-s_2.11 graphx.py clean.txt mixed.txt > output

clean, noisy corpuses
clean to keep counts of words in order to decide OOV or IV based on threshold.
mixed - clean+noisy to form bipartite graph on which random walks are performed

"""
from pyspark import SparkContext
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.functions import udf, col
from pyspark.sql.types import *
import sys
import hashlib
from operator import add
import nlp
from collections import OrderedDict
import __builtin__


NGRAM = 3
THRESHOLD_COUNT = 6
WORD_TYPES = ["OOV","IV"]
sc = SparkContext()
sqlContext = SQLContext(sc)
#Getting Word count map to use
lines = sqlContext.read.text(sys.argv[1]).rdd.map(lambda r: r[0])
countsrdd = lines.flatMap(lambda x: x.split(' ')).map(lambda x: (x, 1)).reduceByKey(add)
# counts = countsrdd.collectAsMap()


def hashed(str):
	return hashlib.sha256(str).hexdigest()


def parsevertices(r):
	r = r.split("|")
	r[0] = str(r[0])
	r[1] = str(r[1])
	# word_count = 0
	# try:
	# 	word_count = counts[r[0]]
	# except:
	# 	word_count = 0
	# word_type = WORD_TYPES[0]
	# if word_count < THRESHOLD_COUNT:
	# 	word_type = WORD_TYPES[1]
	final_list = [(hashed(r[0]), r[0],"word"), (hashed(r[1]), r[1],"context")]
	return final_list

def parseedges(r):
	r = r.split("|")
	r[0] = str(r[0])
	r[1] = str(r[1])
	# final_list = [(vertices_df_dictionary[r[0]], vertices_df_dictionary[r[1]], 1)]
	final_list = [(hashed(r[0]), hashed(r[1]), 1),(hashed(r[1]), hashed(r[0]), 1)]
	return final_list


def create_edge_list(line):
	edgelist = []
	line = line.strip()
	# line = line.decode("utf-8")
	line = line.encode('ascii','ignore')
	line = nlp.clean(line)
	ngramlist = nlp.extract_ngrams(line,NGRAM)
	for ngram in ngramlist:
		ngram = list(ngram)
		index = NGRAM/2
		word = ngram[index]
		ngram[index] = "*"
		edgelist.append(word + "|" + " ".join(ngram))
	return edgelist



mixed_input = sqlContext.read.text(sys.argv[2]).rdd.map(lambda r: r[0])
edge_list_file = mixed_input.flatMap(lambda r: create_edge_list(r))
vertices = edge_list_file.flatMap(lambda r: parsevertices(r)) # returns an RDD list of (id, vertex, type) by spitting the input line

vertices_df = sqlContext.createDataFrame(vertices,["id","value", "type_initial"]).dropDuplicates()



def get_word_type(_type, count):
	if _type == "context":
		return _type
	if count < THRESHOLD_COUNT:
		return "OOV"
	else:
		return "IV"

def getlcscost(n,m):
	lcs_n_m = len(nlp.lcs(n,m))
	max_length = __builtin__.max(len(n),len(m))
	lcsr_n_m = float(lcs_n_m)/float(max_length)
	edit_n_m = nlp.editex("".join(OrderedDict.fromkeys(n)),"".join(OrderedDict.fromkeys(m)))
	if edit_n_m !=0:
		sim_cost_n_m = lcsr_n_m/edit_n_m
		return sim_cost_n_m
	return 0


def get_pagerank(v,n):
	r = g.pageRank(resetProbability=0.3, maxIter=2, sourceId=v)
	c = r.vertices.filter(r.vertices.type == 'IV')
	d = c.rdd.map(lambda x:(x.value, (x.pagerank + getlcscost(n,x.value)))).top(2,key = lambda x: x[1])
	print d


counts_df = sqlContext.createDataFrame(countsrdd, ["value", "count"])
vertices_counts = vertices_df.join(counts_df, vertices_df.value == counts_df.value, "left_outer")
word_type_function = udf(get_word_type, StringType())
vertices_counts = vertices_counts.withColumn("type", word_type_function("type_initial","count")).select("id", vertices_df["value"], "type")
vertices_counts.filter(vertices_counts.type != "context").show(n =100)
print "no of rows are ",vertices_counts.count()


# vertices_df = vertices_df.withColumn("id", monotonically_increasing_id())
# vertices_df_dictionary = vertices_df.rdd.flatMap(lambda r: {r["value"]: r["id"]}.items())
# vertices_df_dictionary = dict(vertices_df_dictionary.collect()) # creates the dictionary to search the ids of vertices while adding to the edges dataframe

edges = edge_list_file.flatMap(lambda r: parseedges(r)) #creates an rdd list of (src,dst, weight)
edges_df = sqlContext.createDataFrame(edges,["src", "dst", "weight"])

edges_df = edges_df.groupBy(["src", "dst"]).sum("weight") #aggregating the weights
edges_df = edges_df.withColumnRenamed("sum(weight)", "weight")

edges_df.show()
print "no of edges are ", edges_df.count()
# vertices_df.show()
print "no of rows are ", vertices_df.count()


from graphframes import *
g = GraphFrame(vertices_counts, edges_df)
g1 = g.outDegrees
# g1 = g1.filter(g1.inDegree == 2)
g1.groupBy("outDegree").count().show()
print g1.count()


OOV_vertices = g.vertices.filter(g.vertices.type == 'OOV').collect()
print "length of oov vertices ", len(OOV_vertices)
i = 0
for vertex in OOV_vertices:
	print i, vertex.value
	i = i + 1
	get_pagerank(vertex.id,vertex.value)

