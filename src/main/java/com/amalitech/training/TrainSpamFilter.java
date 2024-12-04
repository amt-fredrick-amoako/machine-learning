package com.amalitech.training;

import com.amalitech.model.ModelConfiguration;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

public class TrainSpamFilter {
    public static void main(String[] args) {
        MultiLayerConfiguration multiLayerConfiguration = ModelConfiguration.createMultiLayerConfiguration();
        MultiLayerNetwork model =  new MultiLayerNetwork(multiLayerConfiguration);
        model.init();

        // Generate synthetic training data
        double[][] featureData = new double[][]{
                {1, 0, 0, 1, 0}, // Example email features
                {0, 1, 1, 0, 0},
        };
        double[][] labels = new double[][]{
                {1}, // Spam
                {0}, // Not spam
        };

        DataSet trainingData = new DataSet(Nd4j.create(featureData), Nd4j.create(labels));
        DataSetIterator iterator = new ListDataSetIterator<>(trainingData.asList());

        // Train the model
        for (int i = 0; i < 10; i++) { // 10 epochs
            model.fit(iterator);
        }

        System.out.println("Model Trained!");
    }

}

