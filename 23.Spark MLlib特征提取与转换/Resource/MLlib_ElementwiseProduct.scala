package com.xunfang.spark.SparkMLLib.FeatureExtractionAndTransformation

import org.apache.spark.mllib.feature.ElementwiseProduct
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by hjr on 17-7-18.
  */
object MLlib_ElementwiseProduct {
  def main(args: Array[String]): Unit = {
    /**
      *
      */
    val conf = new SparkConf().setMaster("local[4]").setAppName("MLlib_ElementwiseProduct")
    val sc = new SparkContext(conf)

    /**
      * 创建一些矢量数据;也适用于稀疏向量
      */
    val data = sc.parallelize(Array(Vectors.dense(1.0, 2.0, 3.0), Vectors.dense(4.0, 5.0, 6.0)))

    val transformingVector = Vectors.dense(0.0, 1.0, 2.0)
    val transformer = new ElementwiseProduct(transformingVector)

    /**
      * 批量转换 每一行的转换给出相同的结果
      */
    val transformedData = transformer.transform(data)
    transformedData.foreach(println)

    val transformedData2 = data.map(x => transformer.transform(x))
    transformedData2.foreach(println)

    sc.stop()
  }
}
