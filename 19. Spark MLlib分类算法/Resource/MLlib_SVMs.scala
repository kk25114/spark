package com.xunfang.spark.SparkMLLib.Classification

import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by hjr on 17-7-12.
  */
object MLlib_SVMs {
  def main(args: Array[String]): Unit = {

    /**
      * 初始化
      */
    //val conf = new SparkConf().setAppName("MLlib_SVMs").setMaster("spark://10.2.8.11:7077")
    val conf = new SparkConf().setMaster("local[4]").setAppName("MLlib_SVMs")
    val sc = new SparkContext(conf)

    /**
      * 加载数据集
      */
    val data = MLUtils.loadLibSVMFile(sc, "Resource/MLLibData/sample_libsvm_data.txt")

    /**
      * 拆分数据集（60%训练数据集 40%测试数据集）
      */
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    /**
      * 运行训练算法，构建模型
      */
    val numIterations = 100
    val model = SVMWithSGD.train(training, numIterations)

    /**
      * 清除默认的阈值/临界值
      */
    model.clearThreshold()

    /**
      * 计算测试集的原始分数
      */
    val scoreAndLabels = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    /**
      * 评价指标
      * Computes the area under the receiver operating characteristic (ROC) curve
      */
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    println("Area under ROC = " + auROC)

    /**
      * 保存和加载模型
      */
    model.save(sc, "Resource/MLLibData/SVMModel")
    val sameModel = SVMModel.load(sc, "Resource/MLLibData/SVMModel")

    sc.stop()
  }
}
