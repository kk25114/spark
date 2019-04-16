package com.xunfang.spark.SparkMLLib.FeatureExtractionAndTransformation

import org.apache.spark.mllib.feature.Normalizer
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by hjr on 17-7-18.
  */
object MLlib_Normalizer {
  def main(args: Array[String]): Unit = {
    /**
      * 初始化
      */
    val conf = new SparkConf().setMaster("local[4]").setAppName("MLlib_Normalizer")
    val sc = new SparkContext(conf)
    /**
      * 加载数据
      */
    val data = MLUtils.loadLibSVMFile(sc, "Resource/MLLibData/sample_libsvm_data.txt")

    val normalizer1 = new Normalizer()
    val normalizer2 = new Normalizer(p = Double.PositiveInfinity)

    /**
      * 在data1数据里面的每一次采样都将使用L2范式正规化
      */
    val data1 = data.map(x => (x.label, normalizer1.transform(x.features)))
    data1.take(1).foreach(println)

    /**
      * 在data2数据里面的每一次采样都使用L-infty范式正规化
     */
    val data2 = data.map(x => (x.label, normalizer2.transform(x.features)))
    data2.take(1).foreach(println)

    sc.stop()
  }
}
