package com.xunfang.spark.SparkMLLib.Clustering

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by hjr on 17-7-17.
  */
object MLlib_KMeans {
  def main(args: Array[String]): Unit = {
    /**
      * 初始化
      */
    val conf = new SparkConf().setMaster("local[4]").setAppName("MLlib_KMeans")
    val sc = new SparkContext(conf)
    /**
      * 加载解析数据
      */
    val data = sc.textFile("Resource/MLLibData/KMeans/KMeansData/kmeans_data.txt")
    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

    /**
      * 用KMeans算法将数据聚类到两个类簇中去
      */
    val numClusters = 2
    val numIterations = 20
    val clusters = KMeans.train(parsedData, numClusters, numIterations)

    /**
      * 计算误差平方和
      */
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)

    /**
      * 保存和加载模型
      */
    clusters.save(sc, "Resource/MLLibData/KMeans/KMeansModel")
    val sameModel = KMeansModel.load(sc, "Resource/MLLibData/KMeans/KMeansModel")

    sc.stop()
  }
}
