import scala.io.Source
import org.apache.spark._
import org.apache.spark.rdd.RDD
// import classes required for using GraphX
import org.apache.spark.graphx._
import scala.collection.mutable.ArrayBuffer


val filename = "edgelist"

var vertexArray = ArrayBuffer[(VertexId,(String,Int))]()
var VertexId :Long = 1L
var EdgeArray = ArrayBuffer[Edge[Int]]()
var vertex1 = 0L
var vertex2 = 0L
// var vertexpair = Array[(String,String)]();
for(line <- Source.fromFile(filename).getLines()){
  // println(line)
  // DebugParam("vertexArray", vertexArray,"vertexpair",vertexpair,"line",line)
  var flag1 = 0
  var flag2 = 0
  var vertexpair = line.split("\\|")
  // vertexpair.foreach(println)
  //Vertex 1
  if(vertexArray.find(_._2._1 == vertexpair(0)) == None)
  {
    // println("New vertex 1")
    VertexId += 1
    var pair = (VertexId,(vertexpair(0),1))
    vertexArray += pair
    vertex1 = VertexId
  }
  else
  {
    flag1 = 1
    vertex1 = vertexArray.find(_._2._1 == vertexpair(0)).get._1
  }
  //Vertex 2
  if(vertexArray.find(_._2._1 == vertexpair(1)) == None)
  {
    // println("new vertex 2")
    VertexId +=1
    var pair = (VertexId,(vertexpair(1),1))
    vertexArray += pair
    vertex2 = VertexId
  }
  else
  {
    flag2 = 1
    vertex2 = vertexArray.find(_._2._1 == vertexpair(1)).get._1
  }

  //Edge Array
  if(flag1 == 1 && flag2 == 1 && EdgeArray.find(x=>(x.srcId == vertex1 && x.dstId == vertex2))!= None)
  {
    // println("Incrementing edge")
    var element = EdgeArray.find(x=>(x.srcId == vertex1 && x.dstId == vertex2)).get
    var index = EdgeArray.indexOf(element)
    element.attr += 1
    EdgeArray(index) = element

  }
  else
  {
    // println("added edge")
    var edgeele = new Edge(vertex1,vertex2,1)
    EdgeArray +=edgeele
  }

}

//Graph formation
val vertexRDD: RDD[(Long, (String, Int))] = sc.parallelize(vertexArray)
val edgeRDD: RDD[Edge[Int]] = sc.parallelize(EdgeArray)
val graph: Graph[(String, Int), Int] = Graph(vertexRDD, edgeRDD)
for (triplet <- graph.triplets.filter(t => t.attr > 5).collect) {
  println(s"${triplet.srcAttr._1} loves ${triplet.dstAttr._1}")
}
