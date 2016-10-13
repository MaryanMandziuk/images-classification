/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cnn.image.classification;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import org.nd4j.linalg.lossfunctions.LossFunctions;



import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;

//import org.datavec.image.transform.ImageTransform;
//import org.datavec.image.transform.MultiImageTransform;
//import org.datavec.image.transform.ShowImageTransform;

import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;


import java.io.File;
import java.util.Random;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

/**
 *
 * @author maryan
 */
public class CNNImageClassification {

    /**
     * @param args the command line arguments
     */
    
    protected static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    
    public static void main(String[] args) {
          int nChannels = 3;
        int outputNum = 10;
//        int numExamples = 80;
        int batchSize = 10;
        int nEpochs = 20;
        int iterations = 1;
        int seed = 123;
        int height = 32;
        int width = 32;
        Random randNumGen = new Random(seed);
        System.out.println("Load data....");

        
        
        File parentDir = new File("train1/");

        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);


        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);

        //Split the image files into train and test. Specify the train test split as 80%,20%
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 100, 0);
        InputSplit[] filesInDirSplitTest = filesInDir.sample(pathFilter, 0, 100);
        
        
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplitTest[1];
        
        
        System.out.println("train = " + trainData.length());
        System.out.println("test = " + testData.length());
        //Specifying a new record reader with the height and width you want the images to be resized to.
        //Note that the images in this example are all of different size
        //They will all be resized to the height and width specified below
        ImageRecordReader recordReader = new ImageRecordReader(height,width,nChannels,labelMaker);

        //Often there is a need to transforming images to artificially increase the size of the dataset

        recordReader.initialize(trainData);
        
        DataSetIterator dataIterTrain = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
//        recordReader.reset();
        recordReader.initialize(testData);
        DataSetIterator dataIterTest = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
        
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        
        dataIterTrain.setPreProcessor(scaler);
        dataIterTest.setPreProcessor(scaler);
  
        System.out.println("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(true).l2(0.0005)
//                .dropOut(0.5)
                .learningRate(0.001)//.biasLearningRate(0.02)
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation("identity")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .nOut(50)
                        .activation("identity")
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation("relu")
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .setInputType(InputType.convolutional(height, width,nChannels)) //See note below
                .backprop(true).pretrain(false);
        
        
                MultiLayerConfiguration b = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(false).l2(0.005) // tried 0.0001, 0.0005
                .learningRate(0.0001) // tried 0.00001, 0.00005, 0.000001
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(50) // tried 10, 20, 40, 50
                        .activation("relu")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .nOut(100) // tried 25, 50, 100
                        .activation("relu")
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation("relu")
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false)
                .cnnInputSize(height, width, nChannels).build();

                

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        
        

        System.out.println("Train model....");
        model.setListeners(new ScoreIterationListener(1));
//        for( int i=0; i<nEpochs; i++ ) {
//            model.setListeners(new HistogramIterationListener(1));

            MultipleEpochsIterator trainIter = new MultipleEpochsIterator(nEpochs, dataIterTrain, 2);
            model.fit(trainIter);
            
//            System.out.println("*** Completed epoch - " + i + "  ***");

            System.out.println("Evaluate model....");
//            Evaluation eval = new Evaluation(outputNum);
//            while(dataIterTest.hasNext()){
//                DataSet ds = dataIterTest.next();
//                INDArray output = model.output(ds.getFeatureMatrix(), false);
//                eval.eval(ds.getLabels(), output);
//            }
//            System.out.println(eval.stats());
//            dataIterTest.reset();
//        }
        
        

         Evaluation eval1 =model.evaluate(dataIterTest);
         System.out.println(eval1.stats());
         
         System.out.println("****************Example finished********************");
    }
    
}
