package com.xunfang.spark.SparkMLLib.Regression

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by hjr on 17-7-12.
  */
object MLlib_LinearRegression {
  def main(args: Array[String]): Unit = {
    /**
      * 初始化
      */
    //val conf = new SparkConf().setAppName("MLlib_LinearRegression").setMaster("spark://10.2.8.11:7077")
    val conf = new SparkConf().setMaster("local[4]").setAppName("MLlib_LinearRegression")
    val sc = new SparkContext(conf)

    val sqlContext = new SQLContext(sc)

    /**
      * 加载训练数据
      */
    val training = sqlContext.read.format("libsvm")
      .load("Resource/MLLibData/sample_linear_regression_data.txt")

    val lr = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    /**
      * 拟合模型
      */
    val lrModel = lr.fit(training)

    /**
      * 打印回归系数、截距
      */
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // Summarize the model over the training set and print out some metrics
    /**
      * 通过训练集总结模型，打印一些指标
      */
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")

    sc.stop()
  }
}
