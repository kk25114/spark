package com.xunfang.spark.SparkMLLib.Classification

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by hjr on 17-7-12.
  */
object MLlib_MultilayerPerceptronClassifier {
  def main(args: Array[String]): Unit = {
    /**
      * 初始化
      */
    //val conf = new SparkConf().setAppName("MLlib_MultilayerPerceptronClassifier").setMaster("spark://10.2.8.11:7077")
    val conf = new SparkConf().setMaster("local[4]").setAppName("MLlib_MultilayerPerceptronClassifier")
    val sc = new SparkContext(conf)

    val sqlContext = new SQLContext(sc)

    /**
      * 加载数据 存储为“LIBSVM”格式的DataFrame
      */
    val data = sqlContext.read.format("libsvm")
      .load("Resource/MLLibData/sample_multiclass_classification_data.txt")

    /**
      * 拆分数据集为训练数据集（60%）和测试数据集（40%）
      */
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

    /**
      * specify layers for the neural network:
      * input layer of size 4 (features), two intermediate of size 5 and 4
      * and output of size 3 (classes)
      */
    val layers = Array[Int](4, 5, 4, 3)
    /**
      * 创建训练器，并设置其参数
      */
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)
    /**
      * 训练 MLPC-多层感知器分类器 模型
      */
    val model = trainer.fit(train)
    /**
      * 计算测试集的精度
      */
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("precision")
    println("Precision:" + evaluator.evaluate(predictionAndLabels))

    sc.stop()

  }
}
