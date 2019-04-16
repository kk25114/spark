package com.xunfang.spark.SparkMLLib.Classification

import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}
/**
  * Created by hjr on 17-7-12.
  */
object MLlib_NaiveBayes {
  def main(args: Array[String]): Unit = {
    /**
      * 初始化
      */
    val conf = new SparkConf().setMaster("local[4]").setAppName("MLlib_NaiveBayes")
    val sc = new SparkContext(conf)

    /**
      * 加载解析数据集
      */
    val data = sc.textFile("Resource/MLLibData/NaiveBayesData/sample_naive_bayes_data.txt")
    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }

    /**
      * 拆分数据集（60%训练数据集 40%测试数据集）
      */
    val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0)
    val test = splits(1)

    val model = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")

    val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

    /**
      * 保存和加载模型
      */
    model.save(sc, "Resource/MLLibData/NaiveBayesModel")
    val sameModel = NaiveBayesModel.load(sc, "Resource/MLLibData/NaiveBayesModel")


    sc.stop()
  }
}
