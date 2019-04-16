package com.xunfang.spark.SparkMLLib.EvaluationMetrics.ClassificationModelEvaluation

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by hjr on 17-7-19.
  */
object MLlib_MulticlassClassification {
  def main(args: Array[String]): Unit = {
    /**
      * 初始化
      */
    val conf = new SparkConf().setMaster("local[4]").setAppName("MLlib_MulticlassClassification")
    val sc = new SparkContext(conf)
    /**
      * 加载数据集
      */
    val data = MLUtils.loadLibSVMFile(
      sc, "Resource/MLLibData/EvaluationMetrics/MulticlassClassification/sample_multiclass_classification_data.txt")

    /**
      * 拆分数据集为训练数据集（60%）和测试数据集（40%）
      */
    val Array(training, test) = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    training.cache()

    /**
      * 运行训练算法构建模型
      */
    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(3)
      .run(training)

    /**
      * 在测试集上计算原始分数
      */
    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    /**
      * 实例化多级分类评估对象
      */
    val metrics = new MulticlassMetrics(predictionAndLabels)

    /**
      * 混淆矩阵
      */
    println("Confusion matrix:")
    println(metrics.confusionMatrix)

    /**
      * 综合统计
      */
    val precision = metrics.precision
    val recall = metrics.recall // same as true positive rate
    val f1Score = metrics.fMeasure
    println("Summary Statistics")
    println(s"Precision = $precision")
    println(s"Recall = $recall")
    println(s"F1 Score = $f1Score")

    // 标签的精确度
    val labels = metrics.labels
    labels.foreach { l =>
      println(s"Precision($l) = " + metrics.precision(l))
    }

    /**
      * 标签的召回率
      */
    labels.foreach { l =>
      println(s"Recall($l) = " + metrics.recall(l))
    }

    /**
      * 标签的假阳性率（误诊率）
      */
    labels.foreach { l =>
      println(s"FPR($l) = " + metrics.falsePositiveRate(l))
    }

    /**
      * 标签的F值
      */
    labels.foreach { l =>
      println(s"F1-Score($l) = " + metrics.fMeasure(l))
    }

    /**
      * 加权统计
      */
    println(s"Weighted precision: ${metrics.weightedPrecision}")
    println(s"Weighted recall: ${metrics.weightedRecall}")
    println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
    println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")

    sc.stop()
  }
}
