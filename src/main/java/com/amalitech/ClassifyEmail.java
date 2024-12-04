package com.amalitech;

import com.amalitech.model.ModelConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.factory.Nd4j;

public class ClassifyEmail {
    public static void main(String[] args) {
        MultiLayerConfiguration multiLayerConfiguration = ModelConfiguration.createMultiLayerConfiguration();
        MultiLayerNetwork model =  new MultiLayerNetwork(multiLayerConfiguration);
        model.init();


        double[] newEmailFeatures = {1, 0, 1, 0, 1}; // Example feature vector
        double[] output = model.output(Nd4j.create(newEmailFeatures)).toDoubleVector();

        System.out.println("Spam Probability: " + output[0]);
        if (output[0] > 0.5) {
            System.out.println("This is spam!");
        } else {
            System.out.println("This is not spam.");
        }
    }
}

