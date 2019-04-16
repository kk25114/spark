package com.xunfang.spark.SparkMLLib

import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.mllib.util.{KMeansDataGenerator, LinearDataGenerator, LogisticRegressionDataGenerator}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by hjr on 17-7-11.
  */
object MLlib_BasicStatistics {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("MLlib_BasicStatistics").setMaster("local[4]")
    val sc = new SparkContext(conf)

    /**
      * 统计汇总
      */
    val rdd1 = sc.parallelize(Array(Array(1.0, 2.0, 3.0, 4.0), Array(2.0, 3.0, 4.0, 5.0), Array(3.0, 4.0, 5.0, 6.0)))
      .map(f => Vectors.dense(f))
    val observations: RDD[Vector] = rdd1 // an RDD of Vectors

    /**
      * 列统计汇总
      */
    val summary: MultivariateStatisticalSummary = Statistics.colStats(observations)

    // 包含每一列的平均值的稠密向量
    println(summary.mean)
    // 列方差
    println(summary.variance)
    // 每一列的非零数量
    println(summary.numNonzeros)
    // 每一列的L1范数
    println(summary.normL1)
    // 每一列的L2范数
    println(summary.normL2)
    // 每一列的最大值
    println(summary.max)
    // 每一列的最小值
    println(summary.min)

    /**
      * 相关性
      */
    val seriesX: RDD[Double] = sc.parallelize(Array(1.0, 2.0, 3.0, 4.0)) // a series
    val seriesY: RDD[Double] = sc.parallelize(Array(5.0, 6.0, 6.0, 6.0)) // must have the same number of partitions and cardinality as seriesX

    val correlation_pearson: Double = Statistics.corr(seriesX, seriesY, "pearson")
    val correlation_spearman:Double = Statistics.corr(seriesX, seriesY, "spearman")
    println("seriesX and seriesY correlation_pearson（皮尔逊）: "+correlation_pearson)
    println("seriesX and seriesY correlation_spearman(斯皮尔曼): "+correlation_spearman)


    val data: RDD[Vector] = rdd1 // note that each Vector is a row and not a column

    val correlMatrix_pearson: Matrix = Statistics.corr(data, "pearson")
    val correlMatrix_spearman:Matrix = Statistics.corr(data, "spearman")

    println("皮尔逊相关： "+"\n"+correlMatrix_pearson)
    println("斯皮尔曼： "+"\n"+correlMatrix_spearman)

    /**
      * 卡方检验
      */
    val v1 = Vectors.dense(43.0, 9.0)
    val v2 = Vectors.dense(44.0, 4.0)
    val c1 = Statistics.chiSqTest(v1, v2)
    println("卡方检验： "+"\n"+c1)

    /**
      * 生成样本
      */
    val KMeansRDD = KMeansDataGenerator.generateKMeansRDD(sc, 40, 5, 3, 1.0, 2)
    KMeansRDD.count()
    KMeansRDD.take(10).foreach(println)

    val LinearRDD = LinearDataGenerator.generateLinearRDD(sc, 40, 3, 1.0, 2, 0.0)
    LinearRDD.count()
    LinearRDD.take(10).foreach(println)

    val LogisticRDD = LogisticRegressionDataGenerator.generateLogisticRDD(sc, 40, 3, 1.0, 2, 0.5)
    LogisticRDD.count()
    LogisticRDD.take(10).foreach(println)


    sc.stop()
  }
}
