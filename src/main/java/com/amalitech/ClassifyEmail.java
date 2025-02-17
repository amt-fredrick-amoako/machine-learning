package com.amalitech;

import com.amalitech.model.ModelConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.factory.Nd4j;

import java.util.stream.IntStream;


public class ClassifyEmail {
    public static void main(String[] args) {
        MultiLayerConfiguration multiLayerConfiguration = ModelConfiguration.createMultiLayerConfiguration();
        MultiLayerNetwork model = new MultiLayerNetwork(multiLayerConfiguration);
        model.init();

        double[] newEmailFeatures = IntStream.range(0, 1000).mapToDouble(i -> Math.random()).toArray();
        // Populate the feature vector with appropriate values
        // Example: random values

        double[][] input = new double[1][];
        input[0] = newEmailFeatures;

        double[] output = model.output(Nd4j.create(input)).toDoubleVector();

        System.out.println("Spam Probability: " + output[0]);
        if (output[0] > 0.5) {
            System.out.println("This is spam!");
        } else {
            System.out.println("This is not spam.");
        }
    }
}