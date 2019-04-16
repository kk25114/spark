package com.xunfang.spark.SparkMLLib.Regression

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by hjr on 17-7-12.
  */
object MLlib_LinearRegressionWithSGD {
  def main(args: Array[String]): Unit = {
    /**
      * 初始化
      */
    //val conf = new SparkConf().setAppName("MLlib_LinearRegressionWithSGD").setMaster("spark://10.2.8.11:7077")
    val conf = new SparkConf().setMaster("local[6]").setAppName("MLlib_LinearRegressionWithSGD")
    val sc = new SparkContext(conf)

    /**
      * 加载和解析数据
      */
    val data = sc.textFile("Resource/MLLibData/LinearRegressionWithSGD/lpsa.data")
    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }.cache()

    /**
      * 构建模型
      */
    val numIterations = 100
    val stepSize = 0.00000001
    val model = LinearRegressionWithSGD.train(parsedData, numIterations, stepSize)

    /**
      * 评估模型 计算训练误差
      */
    val valuesAndPreds = parsedData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val MSE = valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()
    println("training Mean Squared Error = " + MSE)

    /**
      * 保存和加载模型
      */
    model.save(sc, "Resource/MLLibData/LinearRegressionWithSGD/LinearRegressionModel")
    val sameModel = LinearRegressionModel.load(sc, "Resource/MLLibData/LinearRegressionWithSGD/LinearRegressionModel")

    sc.stop()
  }
}
