package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{IDF, RegexTokenizer, StopWordsRemover}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    /*****************************************/
    /**Pré-traitement des données textuelles**/
    /*****************************************/

    /**On commence par charger les données nettoyées du TP2.**/
    val df_cleaned = spark
      .read
      .load("prepared_trainingset/*.parquet")

    /**2a.
      * 1er stage: On sépare les textes en mots (ou tokens).
      */
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    /**2b.
      * 2e stage: On retire les stop words.
      */
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    /**2c.
      * 3e stage: On convertit nos données textes en vecteurs comptant les tokens
      */
    val vectorizer = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("rawFeatures")

    /**2d.
      * 4e stage: On applique la méthode TD-IDF qui évalue l'importance
      * d'un terme contenu dans un corpus en lui attribuant un poids
      * qui augmente proportionnellement au nombre d'occurences du mot
      * dans le document.
      */
    val idf = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("tfidf")


    /***************************************************/
    /**Conversion des catégories en données numériques**/
    /***************************************************/

    /**3e.
      * 5e stage: On convertit la variable catégorielle “country2”
      * en quantités numériques.
     */
    val indexer_country = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setHandleInvalid("skip")

    /**3f.
      * 6e stage: De même avec currency2.
      */
    val indexer_currency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      .setHandleInvalid("skip")

    /**3g.
      * 7e et 8e stage: Transformer country_indexed et
      * currency_indexed avec un one-hot encoder.
      */
    val encoder_country = new OneHotEncoder()
      .setInputCol("country_indexed")
      .setOutputCol("country_onehot")
    val encoder_currency = new OneHotEncoder()
      .setInputCol("currency_indexed")
      .setOutputCol("currency_onehot")

    /***************************************************/
    /**Adaptation des données pour Spark ML*************/
    /***************************************************/

    /**4h.
      * 9e stage: Assembler les features dans une seule colonne
      * "features".
      */
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa",
      "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("features")

    /**4i.
      * 10e stage: Définition du modèle de régression logistique.
      */
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)

    /**4j.
      * Pipeline: On crée un pipeline qui assemble les 10 stages définis
      * précedemment.
      */
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, vectorizer, idf, indexer_country,
        indexer_currency, encoder_country, encoder_currency, assembler, lr))


    /***************************************************/
    /**Entraînement et tuning du modèle*****************/
    /***************************************************/

    /**5.k
      * Split des données en training et test set 90/10.
      */
    val Array(training, test) = df_cleaned.randomSplit(Array(0.9, 0.1), seed = 12)

    /**5.l
      * Entraînement du classifieur et réglage des hyper-paramètres.
      */
    /* Grille des paramètres à tester*/
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(10e-8, 10e-2, 10e-2))
      .addGrid(vectorizer.minDF, Array(55.0, 95.0, 20.0))
      .build()

    /*Méthode d'évaluation*/
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    /*Train-Validation Split*/
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    /*Lancement du Grid Search sur le set training pour obtenir
    les meilleurs hyper-paramètres.*/
    val model = trainValidationSplit.fit(training)

    /**5.m
      * Test du modèle sur les données test.
      */
    val df_WithPredictions = model.transform(test)
      .select("features", "final_status", "predictions")

    /**5.n
      * Afficher les résultats.
      */
    val f1_score = evaluator.evaluate(df_WithPredictions)
    print("My score: ", f1_score)

    df_WithPredictions.groupBy("final_status", "predictions").count.show()


    /***************************************************/
    /**Sauvegarde du modèle dans un dossier result_TP3**/
    /***************************************************/
    model.write.overwrite().save("./result_TP3")
  }
}

