package com.xunfang.spark.SparkMLLib.Regression

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext

/**
  * Created by hjr on 17-7-12.
  */
object MLlib_GradientBoostedTreeRegression {
  def main(args: Array[String]): Unit = {
    /**
      * 初始化
      */
    //val conf = new SparkConf().setAppName("MLlib_GradientBoostedTreeRegression").setMaster("spark://10.2.8.11:7077")
    val conf = new SparkConf().setMaster("local[8]").setAppName("MLlib_GradientBoostedTreeRegression")
    val sc = new SparkContext(conf)

    val sqlContext = new SQLContext(sc)

    /**
      * 加载解析数据，并强制转化为DataFrame
      */
    val data = sqlContext.read.format("libsvm").load("Resource/MLLibData/sample_libsvm_data.txt")

    /**
      * 自动识别分类特征和索引
      * 设置最大分类 具有大于4个不同的值的特征被视为连续
      */
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)

    /**
      * 拆分数据集为训练数据集（70%）和测试数据集（30%）
      */
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    /**
      * 训练GBT-梯度提升树 模型
      */
    val gbt = new GBTRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")
      .setMaxIter(10)

    /**
      * Chain indexer and GBT in a Pipeline
      */
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, gbt))

    /**
      * 训练模型
      */
    val model = pipeline.fit(trainingData)

    /**
      * 预测
      */
    val predictions = model.transform(testData)

    /**
      * 选择样例行显示
      */
    predictions.select("prediction", "label", "features").show(20)

    /**
      * 选择（预测标签/真实标签）和计算实验误差
      */
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    val gbtModel = model.stages(1).asInstanceOf[GBTRegressionModel]
    println("Learned regression GBT model:\n" + gbtModel.toDebugString)

    sc.stop()
  }
}
