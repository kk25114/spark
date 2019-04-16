package com.xunfang.spark.SparkMLLib.Clustering

import org.apache.spark.mllib.clustering.BisectingKMeans
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by hjr on 17-7-17.
  */
object MLlib_BisectingKMeans {
  def main(args: Array[String]): Unit = {
    /**
      * 初始化
      */
    val conf = new SparkConf().setMaster("local[4]").setAppName("MLlib_BisectingKMeans")
    val sc = new SparkContext(conf)

    /**
      * 加载和解析数据
      */
    def parse(line: String): Vector = Vectors.dense(line.split(" ").map(_.toDouble))
    val data = sc.textFile("Resource/MLLibData/BisectingKMeans/BisectingKMeansData/kmeans_data.txt").map(parse).cache()

    /**
      * 使用二分K均值算法将数据聚类到6个类簇中去
      */
    val bkm = new BisectingKMeans().setK(6)
    val model = bkm.run(data)

    /**
      * 显示计算误差和类簇中心
      */
    println(s"Compute Cost: ${model.computeCost(data)}")
    model.clusterCenters.zipWithIndex.foreach { case (center, idx) =>
      println(s"Cluster Center ${idx}: ${center}")
    }

    sc.stop()
  }
}
