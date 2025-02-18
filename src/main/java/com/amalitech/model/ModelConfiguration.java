package com.amalitech.model;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class ModelConfiguration {
    public static MultiLayerConfiguration createMultiLayerConfiguration(int vocabSize) {
        return new NeuralNetConfiguration.Builder()
                .seed(123) // Set a random seed for reproducibility
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(vocabSize) // Number of input features
                        .nOut(100) // Number of output neurons
                        .activation(Activation.RELU) // Activation function
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .nIn(100) // Input size from the previous layer
                        .nOut(1) // Output size (e.g., spam/not spam)
                        .activation(Activation.SIGMOID) // Sigmoid for binary classification
                        .build())
                .build();
    }
}
