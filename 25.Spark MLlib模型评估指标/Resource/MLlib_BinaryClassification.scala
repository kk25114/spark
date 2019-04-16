package com.xunfang.spark.SparkMLLib.EvaluationMetrics.ClassificationModelEvaluation

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by hjr on 17-7-18.
  */
object MLlib_BinaryClassification {
  def main(args: Array[String]): Unit = {
      /**
        * 初始化
        */
      val conf = new SparkConf().setMaster("local[4]").setAppName("MLlib_BinaryClassification")
      val sc = new SparkContext(conf)

      /**
        * 加载数据
        */
      val data = MLUtils.loadLibSVMFile (
        sc, "Resource/MLLibData/EvaluationMetrics/BinaryClassification/sample_binary_classification_data.txt")

      /**
        * 拆分数据集（60%训练数据集 40%测试数据集）
        */
      val Array(training, test) = data.randomSplit(Array(0.6, 0.4), seed = 11L)
      training.cache()

      /**
        * 运行训练算法，构建模型
        */
      val model = new LogisticRegressionWithLBFGS()
        .setNumClasses(2)
        .run(training)

      /**
        * 明确预测阈值 模型返回原始预测分数
        */
      model.clearThreshold

      /**
        * 计算测试集的原始分数
        */
      val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
        (prediction, label)
      }

      /**
        * 实例化评估对象
        */
      val metrics = new BinaryClassificationMetrics(predictionAndLabels)

      /**
        * 阈值精确率
        */
      val precision = metrics.precisionByThreshold
      precision.foreach { case (t, p) =>
        println(s"Threshold: $t, Precision: $p")
      }

      /**
        * 阈值召回率
        */
      val recall = metrics.recallByThreshold
      recall.foreach { case (t, r) =>
        println(s"Threshold: $t, Recall: $r")
      }

      /**
        * PR曲线
        */
      val PRC = metrics.pr
      PRC.foreach(println)

      /**
        * F 值
        */
      val f1Score = metrics.fMeasureByThreshold
      f1Score.foreach { case (t, f) =>
        println(s"Threshold: $t, F-score: $f, Beta = 1")
      }

      val beta = 0.5
      val fScore = metrics.fMeasureByThreshold(beta)
      f1Score.foreach { case (t, f) =>
        println(s"Threshold: $t, F-score: $f, Beta = 0.5")
      }

      /**
        * AUPRC
        */
      val auPRC = metrics.areaUnderPR
      println("Area under precision-recall curve = " + auPRC)

      /**
        * 通过ROC和PR曲线计算阈值
        */
      val thresholds = precision.map(_._1)
      thresholds.foreach(println)

      /**
        * ROC曲线
        */
      val roc = metrics.roc
      roc.foreach(println)

      /**
        * AUROC
        */
      val auROC = metrics.areaUnderROC
      println("Area under ROC = " + auROC)

      sc.stop()
    }
}
