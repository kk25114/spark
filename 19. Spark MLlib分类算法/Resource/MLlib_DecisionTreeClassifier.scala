package com.xunfang.spark.SparkMLLib.Classification

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by hjr on 17-7-12.
  */
object MLlib_DecisionTreeClassifier {
  def main(args: Array[String]): Unit = {
    /**
      * 初始化
      */
    //val conf = new SparkConf().setAppName("MLlib_DecisionTreeClassifier").setMaster("spark://10.2.8.11:7077")
    val conf = new SparkConf().setMaster("local[4]").setAppName("MLlib_DecisionTreeClassifier")
    val sc = new SparkContext(conf)

    val sqlContext = new SQLContext(sc)

    /**
      * 加载数据 存储为“LIBSVM”格式的DataFrame
      */
    val data = sqlContext.read.format("libsvm").load("Resource/MLLibData/sample_libsvm_data.txt")

    /**
      * 索引标签，将元数据添加到标签列中
      * 适用整个数据集，包括所有的索引标签
      */
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)
    /**
      * 自动识别分类特征和索引
      * 设置最大分类 具有大于4个不同的值的特征被视为连续
      */
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4) // features with > 4 distinct values are treated as continuous
      .fit(data)

    /**
      * 拆分数据集为训练数据集（70%）和测试数据集（30%）
      */
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    /**
      * 训练决策数分类模型
      */
    val dt = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")

    /**
      * 将索引标签强制转化为原始标签
      */
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    /**
      * Chain indexers and tree in a Pipeline
      */
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

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
    predictions.select("predictedLabel", "label", "features").show(20)

    /**
      * 选择（预测标签/真实标签）和计算实验误差
      */
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("precision")
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))

    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
    println("Learned classification tree model:\n" + treeModel.toDebugString)

    
    sc.stop()
  }
}
