"""
spark-submit --packages graphframes:graphframes:0.2.0-spark2.0-s_2.11 graphx.py filename > output



"""
from pyspark import SparkContext
from pyspark.sql import *
from pyspark.sql.functions import monotonically_increasing_id
import sys
from operator import add

THRESHOLD_COUNT = 6
WORD_TYPES = ["OOV","IV"]
sc = SparkContext()
sqlContext = SQLContext(sc)
#Getting Word count map to use
lines = sqlContext.read.text(sys.argv[1]).rdd.map(lambda r: r[0])
counts = lines.flatMap(lambda x: x.split(' ')).map(lambda x: (x, 1)).reduceByKey(add).collectAsMap()


def parsevertices(r):
	r = r.split("|")
	r[0] = str(r[0])
	r[1] = str(r[1])
	word_count = 0
	try:
		word_count = counts[r[0]]
	except:
		word_count = 0
	word_type = WORD_TYPES[0]
	if word_count < THRESHOLD_COUNT:
		word_type = WORD_TYPES[1]
	final_list = [(r[0],word_type), (r[1],"context")]
	return final_list

def parseedges(r, vertices_df):
	r = r.split("|")
	r[0] = str(r[0])
	r[1] = str(r[1])
	# index = [0,0]
	# index[0] = vertices_df.rdd.filter(lambda x: x["value"] == r[0]).collect()[0]["id"]
	# index[1] = vertices_df.rdd.filter(lambda x: x["value"] == r[1]).collect()[0]["id"]
	# print index

	final_list = [(vertices_df_dictionary[r[0]], vertices_df_dictionary[r[1]], 1)]
	# print final_list
	return final_list


edge_list_file = sqlContext.read.text("edgelist").rdd.map(lambda r: r[0])
vertices = edge_list_file.flatMap(lambda r: parsevertices(r)) # returns an RDD list of (vertex, type) by spitting the input line

vertices_df = sqlContext.createDataFrame(vertices,["value", "type"]).dropDuplicates()
vertices_df = vertices_df.withColumn("id", monotonically_increasing_id())

vertices_df_dictionary = vertices_df.rdd.flatMap(lambda r: {r["value"]: r["id"]}.items())
vertices_df_dictionary = dict(vertices_df_dictionary.collect()) # creates the dictionary to search the ids of vertices while adding to the edges dataframe

edges = edge_list_file.flatMap(lambda r: parseedges(r, vertices_df_dictionary)) #creates an rdd list of (src,dst, weight)
edges_df = sqlContext.createDataFrame(edges,["src", "dst", "weight"])

edges_df = edges_df.groupBy(["src", "dst"]).sum("weight") #aggregating the weights
edges_df = edges_df.withColumnRenamed("sum(weight)", "weight")

edges_df.show()
vertices_df.show()

from graphframes import *
g = GraphFrame(vertices_df, edges_df)
g1 = g.inDegrees
g1 = g1.filter(g1.inDegree == 2)
g1.show()
print g1.count()


# filtered = vertices_df.rdd.filter(lambda x: x["value"] == "on").collect()
# print filtered
# print edges.collect()


# v = sqlContext.createDataFrame([
#   ("a", "Alice", 34),
#   ("b", "Bob", 36),
#   ("c", "Charlie", 30),
# ], ["id", "name", "age"])
# # Create an Edge DataFrame with "src" and "dst" columns
# e = sqlContext.createDataFrame([
#   ("a", "b", "friend"),
#   ("b", "c", "follow"),
#   ("c", "b", "follow"),
# ], ["src", "dst", "relationship"])
# # Create a GraphFrame
# from graphframes import *
# g = GraphFrame(v, e)

# # Query: Get in-degree of each vertex.
# g.inDegrees.show()

# # Query: Count the number of "follow" connections in the graph.
# g.edges.filter("relationship = 'follow'").count()

# # Run PageRank algorithm, and show results.
# results = g.pageRank(resetProbability=0.01, maxIter=20)
# results.vertices.select("id", "pagerank").show()
