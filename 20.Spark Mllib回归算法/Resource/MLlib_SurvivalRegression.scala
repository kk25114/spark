package com.xunfang.spark.SparkMLLib.Regression

import org.apache.spark.ml.regression.AFTSurvivalRegression
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext

/**
  * Created by hjr on 17-7-12.
  */
object MLlib_SurvivalRegression {
  def main(args: Array[String]): Unit = {
    /**
      * 初始化
      */
    //val conf = new SparkConf().setAppName("MLlib_SurvivalRegression").setMaster("spark://10.2.8.11:7077")
    val conf = new SparkConf().setMaster("local[4]").setAppName("MLlib_SurvivalRegression")
    val sc = new SparkContext(conf)

    val sqlContext = new SQLContext(sc)
    /**
      * 创建训练数据
      */
    val training = sqlContext.createDataFrame(Seq(
      (1.218, 1.0, Vectors.dense(1.560, -0.605)),
      (2.949, 0.0, Vectors.dense(0.346, 2.158)),
      (3.627, 0.0, Vectors.dense(1.380, 0.231)),
      (0.273, 1.0, Vectors.dense(0.520, 1.151)),
      (4.199, 0.0, Vectors.dense(0.795, -0.226))
    )).toDF("label", "censor", "features")
    val quantileProbabilities = Array(0.3, 0.6)
    val aft = new AFTSurvivalRegression()
      .setQuantileProbabilities(quantileProbabilities)
      .setQuantilesCol("quantiles")

    val model = aft.fit(training)

    /**
      * 打印精度、截距
      * Print the coefficients, intercept and scale parameter for AFT survival regression
      */
    println(s"Coefficients: ${model.coefficients} Intercept: " +
      s"${model.intercept} Scale: ${model.scale}")
    model.transform(training).show(false)
	
	sc.stop()
  }
}
