package com.amalitech;

import com.amalitech.model.ModelConfiguration;
import com.amalitech.model.SentimentModel;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class SentimentAnalysis {
    public static void main(String[] args) throws Exception {
        int vocabSize = 1000; // Adjust as needed
        File modelFile = new File("src/main/resources/sentiments.zip");
        File trainingDataFile = new File("src/main/resources/emails.csv");

        // Load or train the sentiment model

        SentimentModel sentimentModel = SentimentModel.loadOrTrain(modelFile, trainingDataFile, vocabSize);

        // Read user input and perform sentiment analysis
        Scanner scanner = new Scanner(System.in);
        System.out.println("Enter text for sentiment analysis:");
        String inputText = scanner.nextLine();

        double sentimentProb = sentimentModel.analyzeSentiment(inputText);
        System.out.println("Sentiment probability: " + sentimentProb);
        if (sentimentProb > 0.5) {
            System.out.println("This is spam!");
        } else {
            System.out.println("This is not spam.");
        }
    }
}
