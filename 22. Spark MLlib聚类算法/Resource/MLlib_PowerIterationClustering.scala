package com.xunfang.spark.SparkMLLib.Clustering

import org.apache.spark.mllib.clustering.{PowerIterationClustering, PowerIterationClusteringModel}

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by hjr on 17-7-17.
  */
object MLlib_PowerIterationClustering {
  def main(args: Array[String]): Unit = {
    /**
      * 初始化
      */
    val conf = new SparkConf().setMaster("local[4]").setAppName("MLlib_PowerIterationClustering")
    val sc = new SparkContext(conf)

    /**
      * 加载和解析数据
      */
    val data = sc.textFile("Resource/MLLibData/PowerIterationClustering/PCI_Data/pic_data.txt")
    val similarities = data.map { line =>
      val parts = line.split(' ')
      (parts(0).toLong, parts(1).toLong, parts(2).toDouble)
    }

    /**
      * 使用PIC算法将数据聚类到两个类簇中
      */
    val pic = new PowerIterationClustering()
      .setK(2)
      .setMaxIterations(10)
    val model = pic.run(similarities)

    model.assignments.foreach { a =>
      println(s"${a.id} -> ${a.cluster}")
    }

    /**
      * 保存和加载模型
      */
    model.save(sc, "Resource/MLLibData/PowerIterationClustering/PCI_Model")
    val sameModel = PowerIterationClusteringModel.load(sc, "Resource/MLLibData/PowerIterationClustering/PCI_Model")

    sc.stop()
  }
}
