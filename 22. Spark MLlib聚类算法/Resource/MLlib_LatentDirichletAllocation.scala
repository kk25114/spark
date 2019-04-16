package com.xunfang.spark.SparkMLLib.Clustering

import org.apache.spark.mllib.clustering.{LDA, DistributedLDAModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by hjr on 17-7-17.
  */
object MLlib_LatentDirichletAllocation {
  def main(args: Array[String]): Unit = {
    /**
      * 初始化
      */
    val conf = new SparkConf().setMaster("local[4]").setAppName("MLlib_LatentDirichletAllocation")
    val sc = new SparkContext(conf)

    /**
      * 加载和解析数据
      */
    val data = sc.textFile("Resource/MLLibData/LatentDirichletAllocation/LDA_Data/sample_lda_data.txt")
    val parsedData = data.map(s => Vectors.dense(s.trim.split(' ').map(_.toDouble)))
    /**
      * 使用唯一的ID检索文档
      */
    val corpus = parsedData.zipWithIndex.map(_.swap).cache()

    /**
      * 使用LDA将文档聚类成3个topics
      */
    val ldaModel = new LDA().setK(3).run(corpus)

    /**
      * 输出topics
      */
    println("Learned topics (as distributions over vocab of " + ldaModel.vocabSize + " words):")
    val topics = ldaModel.topicsMatrix
    for (topic <- Range(0, 3)) {
      print("Topic " + topic + ":")
      for (word <- Range(0, ldaModel.vocabSize)) { print(" " + topics(word, topic)); }
      println()
    }

    /**
      * 保存和加载模型
      */
    ldaModel.save(sc, "Resource/MLLibData/LatentDirichletAllocation/LDA_Model")
    val sameModel = DistributedLDAModel.load(sc, "Resource/MLLibData/LatentDirichletAllocation/LDA_Model")

    sc.stop()
  }
}
