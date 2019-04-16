package com.xunfang.spark.SparkMLLib.FeatureExtractionAndTransformation

import org.apache.spark.mllib.feature.{StandardScaler, StandardScalerModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by hjr on 17-7-18.
  */
object MLlib_StandardScaler {
  def main(args: Array[String]): Unit = {
    /**
      * 初始化
      */
    val conf = new SparkConf().setMaster("local[4]").setAppName("MLlib_StandardScaler")
    val sc = new SparkContext(conf)
    /**
      * 加载数据
      */
    val data = MLUtils.loadLibSVMFile(sc, "Resource/MLLibData/sample_libsvm_data.txt")

    val scaler1 = new StandardScaler().fit(data.map(x => x.features))
    val scaler2 = new StandardScaler(withMean = true, withStd = true).fit(data.map(x => x.features))

    val scaler3 = new StandardScalerModel(scaler2.std, scaler2.mean)

    // data1 单元方差
    val data1 = data.map(x => (x.label, scaler1.transform(x.features)))
    data1.take(1).foreach(println)
    /**
      *  没有强制转换特征到密度向量，
      *  零均值的转换将增加稀疏向量的异常
      */
    // data2 单元方差 零均值
    val data2 = data.map(x => (x.label, scaler2.transform(Vectors.dense(x.features.toArray))))
    data2.take(1).foreach(println)

    sc.stop()
  }
}
