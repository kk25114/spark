package com.xunfang.spark.SparkMLLib.Classification

import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by hjr on 17-7-12.
  */
object MLlib_LogisticRegression {
  def main(args: Array[String]): Unit = {
    /**
      * 初始化
      */
    //val conf = new SparkConf().setAppName("MLlib_LogisticRegression").setMaster("spark://10.2.8.11:7077")
    val conf = new SparkConf().setMaster("local[4]").setAppName("MLlib_LogisticRegression")
    val sc = new SparkContext(conf)

    /**
      * 加载数据集
      */
    val data = MLUtils.loadLibSVMFile(sc, "Resource/MLLibData//sample_libsvm_data.txt")

    /**
      * 拆分数据集（60%训练数据集 40%测试数据集）
      */
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    /**
      * 运行训练算法，构建模型
      */
    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(10)
      .run(training)

    /**
      * 计算测试集的原始分数
      */
    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    /**
      * 评价指标
      */
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val precision = metrics.precision
    println("Precision = " + precision)

    /**
      * 保存和加载模型
      */
    model.save(sc, "Resource/MLLibData/LogisticRegressionModel")
    val sameModel = LogisticRegressionModel.load(sc, "Resource/MLLibData/LogisticRegressionModel")

    sc.stop()
  }
}