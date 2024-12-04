package com.amalitech.model;
//
//import org.deeplearning4j.nn.api.OptimizationAlgorithm;
//import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
//import org.deeplearning4j.nn.conf.layers.DenseLayer;
//import org.deeplearning4j.nn.conf.layers.OutputLayer;
//import org.deeplearning4j.nn.weights.WeightInit;
//import org.nd4j.linalg.lossfunctions.LossFunctions;
//import org.nd4j.linalg.activations.Activation;
//import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
//
//public class SpamFilterModel {
//    public static void main(String[] args) {
//        MultiLayerConfiguration config = new MultiLayerConfiguration.Builder()
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .weightInit(WeightInit.XAVIER)
//                .list()
//                .layer(new DenseLayer.Builder()
//                        .nIn(1000) // Number of features
//                        .nOut(100) // Hidden layer size
//                        .activation(Activation.RELU)
//                        .build())
//                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
//                        .activation(Activation.SIGMOID)
//                        .nIn(100)
//                        .nOut(1) // Binary classification
//                        .build())
//                .build();
//
//        MultiLayerNetwork model = new MultiLayerNetwork(config);
//        model.init();
//
//        System.out.println("Spam Filter Model Ready!");
//    }
//}

