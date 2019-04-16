package com.xunfang.spark.SparkMLLib.Regression

import org.apache.spark.mllib.regression.{IsotonicRegression, IsotonicRegressionModel}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by hjr on 17-7-13.
  */
object MLlib_IsotonicRegression {
  def main(args: Array[String]): Unit = {
    /**
      * 初始化
      */
    //val conf = new SparkConf().setAppName("MLlib_IsotonicRegression").setMaster("spark://10.2.8.11:7077")
    val conf = new SparkConf().setMaster("local[6]").setAppName("MLlib_IsotonicRegression")
    val sc = new SparkContext(conf)

    val sqlContext = new SQLContext(sc)
    val data = sc.textFile("Resource/MLLibData/sample_isotonic_regression_data.txt")

    /**
      * 从输入数据创建tuple（标签，特征，权重）元组，输入数据默认权重设置为1.0
      */
    val parsedData = data.map { line =>
      val parts = line.split(',').map(_.toDouble)
      (parts(0), parts(1), 1.0)
    }

    /**
      * 拆分数据集为训练数据集（60%）和测试数据集（40%）
      */
    val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0)
    val test = splits(1)

    /**
      * 从训练数据中创建一个保序回归模型
      */
    val model = new IsotonicRegression().setIsotonic(true).run(training)

    /**
      * 创建一个（预测标签，真实标签）的tuple元组
      */
    val predictionAndLabel = test.map { point =>
      val predictedLabel = model.predict(point._2)
      (predictedLabel, point._1)
    }

    /**
      * 计算预测值/标签 与真实值/标签 之间的均方误差
      */
    val meanSquaredError = predictionAndLabel.map { case (p, l) => math.pow((p - l), 2) }.mean()
    println("Mean Squared Error = " + meanSquaredError)

    /**
      * 保存和加载模型
      */
    model.save(sc, "Resource/MLLibData/IsotonicRegressionModel")
    val sameModel = IsotonicRegressionModel.load(sc, "Resource/MLLibData/IsotonicRegressionModel")

    sc.stop()
  }
}
