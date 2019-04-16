package com.xunfang.spark.SparkMLLib.Clustering

import org.apache.spark.mllib.clustering.GaussianMixture
import org.apache.spark.mllib.clustering.GaussianMixtureModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by hjr on 17-7-17.
  */
object MLlib_GaussianMixture {
  def main(args: Array[String]): Unit = {
    /**
      * 初始化
      */
    val conf = new SparkConf().setMaster("local[4]").setAppName("MLlib_GaussianMixture")
    val sc = new SparkContext(conf)

    /**
      * 加载解析数据
      */
    val data = sc.textFile("Resource/MLLibData/GaussianMixture/GaussianMixtureData/gmm_data.txt")
    val parsedData = data.map(s => Vectors.dense(s.trim.split(' ').map(_.toDouble))).cache()

    /**
      高斯混合将数据聚类到两个类簇中去
      */
    val gmm = new GaussianMixture().setK(2).run(parsedData)

    /**
      * 保存和加载模型
      */
    gmm.save(sc, "Resource/MLLibData/GaussianMixture/GaussianMixtureModel")
    val sameModel = GaussianMixtureModel.load(sc, "Resource/MLLibData/GaussianMixture/GaussianMixtureModel")

    /**
      * 极大似然模型的输出参数
      */
    for (i <- 0 until gmm.k) {
      println("weight=%f\nmu=%s\nsigma=\n%s\n" format
        (gmm.weights(i), gmm.gaussians(i).mu, gmm.gaussians(i).sigma))
    }

    sc.stop()
  }
}
