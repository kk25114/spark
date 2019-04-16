package com.xunfang.spark.SparkMLLib.FeatureExtractionAndTransformation

import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by hjr on 17-7-18.
  */
object MLlib_Word2Vec {
  def main(args: Array[String]): Unit = {
    /**
      * 初始化
      */
    val conf = new SparkConf().setMaster("local[4]").setAppName("MLlib_Word2Vec")
    val sc = new SparkContext(conf)

    /**
      * 拆分数据
      */
    val input = sc.textFile("Resource/MLLibData/FeatureExtractionAndTransformation/Word2VecData/text8")
      .map(line => line.split(" ").toSeq)

    val word2vec = new Word2Vec()

    val model = word2vec.fit(input)
    // 寻找一个单词的同义词
    val synonyms = model.findSynonyms("china", 40)

    for((synonym, cosineSimilarity) <- synonyms) {
      println(s"$synonym $cosineSimilarity")
    }

    /**
      * 保存和加载模型
      */
    model.save(sc, "Resource/MLLibData/FeatureExtractionAndTransformation/Word2VecModel")
    val sameModel = Word2VecModel.load(sc, "Resource/MLLibData/FeatureExtractionAndTransformation/Word2VecModel")

    sc.stop()
  }
}
