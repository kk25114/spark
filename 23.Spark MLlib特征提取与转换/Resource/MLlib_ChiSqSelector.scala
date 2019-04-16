package com.xunfang.spark.SparkMLLib.FeatureExtractionAndTransformation

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.feature.ChiSqSelector
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by hjr on 17-7-18.
  */
object MLlib_ChiSqSelector {
  def main(args: Array[String]): Unit = {
    /**
      * 初始化
      */
    val conf = new SparkConf().setMaster("local[4]").setAppName("MLlib_ChiSqSelector")
    val sc = new SparkContext(conf)

    /**
      * 加载数据
      */
    val data = MLUtils.loadLibSVMFile(sc, "Resource/MLLibData/sample_libsvm_data.txt")
    /**
      * 离散化数据
      */
    val discretizedData = data.map { lp =>
      LabeledPoint(lp.label, Vectors.dense(lp.features.toArray.map { x => (x / 16).floor } ) )
    }
    discretizedData.take(1).foreach(println)

    // 创建一个卡方选择器，选择692个特征的前50个
    val selector = new ChiSqSelector(50)
    // 创建一个卡方选择器模型
    val transformer = selector.fit(discretizedData)
    // 从每一个特征向量过滤前50个特征
    val filteredData = discretizedData.map { lp =>
      LabeledPoint(lp.label, transformer.transform(lp.features))
    }
    filteredData.take(5).foreach(println)

    sc.stop()
  }
}
