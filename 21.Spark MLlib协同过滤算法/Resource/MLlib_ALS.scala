package com.xunfang.spark.SparkMLLib.CollaborativeFiltering

import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by hjr on 17-7-17.
  */
object MLlib_ALS {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local[4]").setAppName("MLlib_ALS")
    val sc = new SparkContext(conf)

    /**
      * 加载解析数据
      */
    val data = sc.textFile("Resource/MLLibData/CollaborativeFiltering/ALS_Data/test.data")
    val ratings = data.map(_.split(',') match { case Array(user, item, rate) =>
      Rating(user.toInt, item.toInt, rate.toDouble)
    })

    /**
      * 使用ALS算法构建推荐模型
      */
    val rank = 10
    val numIterations = 10
    val model = ALS.train(ratings, rank, numIterations, 0.01)

    /**
      * 在训练数据上评估模型
      */
    val usersProducts = ratings.map { case Rating(user, product, rate) =>
      (user, product)
    }
    val predictions =
      model.predict(usersProducts).map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }
    val ratesAndPreds = ratings.map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }.join(predictions)
    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean()
    println("Mean Squared Error = " + MSE)

    /**
      * 保存和加载模型
      */
    model.save(sc, "Resource/MLLibData/CollaborativeFiltering/CollaborativeFilterModel")
    val sameModel = MatrixFactorizationModel.load(sc, "Resource/MLLibData/CollaborativeFiltering/CollaborativeFilterModel")

    sc.stop()
  }
}
