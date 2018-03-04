package arturro.facerecog.dl4j

/*
 * Created by Artur Fejklowicz
 * 2018-03-04
 */

import java.io.File
import org.datavec.api.io.filters.BalancedPathFilter
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.records.listener.impl.LogRecordListener
import org.datavec.api.split.{FileSplit, InputSplit}
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.datavec.image.transform.ResizeImageTransform
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.transferlearning.{FineTuneConfiguration, TransferLearning, TransferLearningHelper}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.util.ModelSerializer
import org.deeplearning4j.zoo.model.VGG16
import org.deeplearning4j.zoo._
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.Logger
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.{DataNormalization, ImagePreProcessingScaler}


object Main {
  def main(args: Array[String]): Unit = {
    val log:Logger = org.slf4j.LoggerFactory.getLogger("arturro.facerecog.dl4j.Main")
    val h = 224
    val w = 224
    val ch = 3
    val seed = 123L
    val randNumGen = new java.util.Random(seed)
    val batchSize = 10
    val outputNum = 21

    println(s"${Console.YELLOW}-------- 1. DATA PREPARATION -------${Console.RESET}")
    val labelMaker:ParentPathLabelGenerator = new ParentPathLabelGenerator()
    val allImagesDir:File = new File("C:\\github\\deeplearning\\resources\\dl4j-vggface\\facesFixed")
    val pathFilter:BalancedPathFilter = new BalancedPathFilter(randNumGen, NativeImageLoader.ALLOWED_FORMATS, labelMaker)
    val allImgsDirSplit:FileSplit = new FileSplit(allImagesDir, NativeImageLoader.ALLOWED_FORMATS,randNumGen)
    val allImgsTrainTestSplit:Array[InputSplit] = allImgsDirSplit.sample(pathFilter, 75, 25)
    val train = allImgsTrainTestSplit(0)
    val test = allImgsTrainTestSplit(1)

    val rit:ResizeImageTransform = new ResizeImageTransform(w, h)

    // train iterator
    val recordReaderTrain:ImageRecordReader = new ImageRecordReader(h,w,ch,labelMaker, rit)
    recordReaderTrain.initialize(train)
    val trainIter:DataSetIterator = new RecordReaderDataSetIterator(recordReaderTrain,batchSize,1,outputNum)
    trainIter.setPreProcessor( new VGG16ImagePreProcessor())

    // test iterator
    val recordReaderTest:ImageRecordReader = new ImageRecordReader(h,w,ch,labelMaker, rit)
    recordReaderTrain.initialize(test)
    val testIter:DataSetIterator = new RecordReaderDataSetIterator(recordReaderTrain,batchSize,1,outputNum)
    testIter.setPreProcessor( new VGG16ImagePreProcessor())

    println(s"${Console.YELLOW}-------- 2. LOAD MODEL FROM INTERNET -------${Console.RESET}")
    val zooModel = new VGG16()
    val pretrainedNet:ComputationGraph = zooModel.initPretrained(PretrainedType.VGGFACE).asInstanceOf[ComputationGraph]
    println(pretrainedNet.summary())

    println(s"${Console.YELLOW}-------- 3. FINE TUNE -------${Console.RESET}")
    val fineTuneConf:FineTuneConfiguration = new FineTuneConfiguration.Builder()
      .learningRate(5e-5)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(Updater.NESTEROVS)
      .seed(seed)
      .build()

    println(s"${Console.YELLOW}-------- 4. MODIFY LAST LAYER -------${Console.RESET}")
    val numClasses = 21
    val vgg16Transfer:ComputationGraph = new TransferLearning.GraphBuilder(pretrainedNet)
      .fineTuneConfiguration(fineTuneConf)
      .setFeatureExtractor("fc7")
      .removeVertexKeepConnections("fc8")
      .addLayer("fc8",
        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
          .nIn(4096).nOut(numClasses)
          .weightInit(WeightInit.XAVIER)
          .activation(Activation.SOFTMAX).build(), "fc7")
      .build()
    println(vgg16Transfer.summary())

    println(s"${Console.YELLOW}-------- 5. EVALUATE BEFORE -------${Console.RESET}")
    var eval = vgg16Transfer.evaluate(testIter)
    println(eval.stats())
    testIter.reset()

    println(s"${Console.YELLOW}-------- 6. TRAIN AND EVALUATE EVERY 10 images -------${Console.RESET}")
    var i = 0
    while(trainIter.hasNext()) {
      println(s"${Console.YELLOW}-------- 6. TRAIN IMAGE $i -------${Console.RESET}")
      vgg16Transfer.fit(trainIter.next())
      if (i % 10 == 0) {
        println(s"${Console.YELLOW}-------- 6. EVALUATE IMAGE $i -------${Console.RESET}")
        println(vgg16Transfer.evaluate(testIter).stats())
        testIter.reset()
      }
      i += 1
    }
    println("Model build complete")
  }
}
