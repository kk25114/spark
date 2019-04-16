package com.xunfang.spark.SparkMLLib.EvaluationMetrics.ClassificationModelEvaluation

import org.apache.spark.mllib.evaluation.MultilabelMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by hjr on 17-7-19.
  */
object MLlib_MultilabelClassification {
  def main(args: Array[String]): Unit = {
    /**
      * 初始化
      */
    val conf = new SparkConf().setMaster("local[4]").setAppName("MLlib_MultilabelClassification")
    val sc = new SparkContext(conf)

    val scoreAndLabels: RDD[(Array[Double], Array[Double])] = sc.parallelize(
      Seq((Array(0.0, 1.0), Array(0.0, 2.0)),
        (Array(0.0, 2.0), Array(0.0, 1.0)),
        (Array.empty[Double], Array(0.0)),
        (Array(2.0), Array(2.0)),
        (Array(2.0, 0.0), Array(2.0, 0.0)),
        (Array(0.0, 1.0, 2.0), Array(0.0, 1.0)),
        (Array(1.0), Array(1.0, 2.0))), 2)

    /**
      * 实例化多标签分类评估对象
      */
    val metrics = new MultilabelMetrics(scoreAndLabels)

    /**
      * 综合统计
      */
    println(s"Recall = ${metrics.recall}")
    println(s"Precision = ${metrics.precision}")
    println(s"F1 measure = ${metrics.f1Measure}")
    println(s"Accuracy = ${metrics.accuracy}")

    /**
      * 单个标签统计
      */
    metrics.labels.foreach(label =>
      println(s"Class $label precision = ${metrics.precision(label)}"))
    metrics.labels.foreach(label => println(s"Class $label recall = ${metrics.recall(label)}"))
    metrics.labels.foreach(label => println(s"Class $label F1-score = ${metrics.f1Measure(label)}"))

    /**
      * 微统计
      */
    println(s"Micro recall = ${metrics.microRecall}")
    println(s"Micro precision = ${metrics.microPrecision}")
    println(s"Micro F1 measure = ${metrics.microF1Measure}")

    /**
      * 汉明损失
      */
    println(s"Hamming loss = ${metrics.hammingLoss}")

    // Subset accuracy
    /**
      * 子集准确率
      */
    println(s"Subset accuracy = ${metrics.subsetAccuracy}")

    sc.stop()
  }
}
