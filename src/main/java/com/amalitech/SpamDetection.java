package com.amalitech;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class SpamDetection {

    public static void main(String[] args) throws Exception {
        // Load and preprocess the dataset
        List<String> emails = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        BasicLineIterator iterator = new BasicLineIterator(new File("src/main/resources/emails.csv"));
        while (iterator.hasNext()) {
            String line = iterator.nextSentence();
            String[] parts = line.split(",", 2);
            labels.add(Integer.parseInt(parts[0].trim()));
            emails.add(parts[1].trim());
        }

        // Tokenize and vectorize the emails
        DefaultTokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        int vocabSize = 1000; // Adjust as needed
        INDArray features = Nd4j.create(emails.size(), vocabSize);
        INDArray labelsArray = Nd4j.create(labels.size(), 1);
        for (int i = 0; i < emails.size(); i++) {
            String email = emails.get(i);
            INDArray emailVector = Nd4j.create(vocabSize);
            // Use lambda to update counts for each token
            tokenizerFactory.create(email).getTokens().forEach(token -> {
                int index = Math.abs(token.hashCode() % vocabSize);
                emailVector.putScalar(index, emailVector.getDouble(index) + 1);
            });
            features.putRow(i, emailVector);
            labelsArray.putScalar(i, labels.get(i));
        }

        DataSet dataSet = new DataSet(features, labelsArray);

        // Normalize the dataset
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(dataSet);
        normalizer.transform(dataSet);

        // Configure and train the model
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(vocabSize)
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .nIn(100)
                        .nOut(1)
                        .activation(Activation.SIGMOID)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        model.setListeners(new ScoreIterationListener(10));
        model.fit(dataSet);

        // Classify a new email
        String newEmail = "Get rich quick! Earn $1000 a day from home. Sign up now!";
        // Create the vector for the new email
        INDArray newEmailVector = Nd4j.create(vocabSize);
        INDArray finalNewEmailVector = newEmailVector;
        tokenizerFactory.create(newEmail).getTokens().forEach(token -> {
            int index = Math.abs(token.hashCode() % vocabSize);
            finalNewEmailVector.putScalar(index, finalNewEmailVector.getDouble(index) + 1);
        });
        // Reshape to [1, vocabSize] to match the training features
        newEmailVector = newEmailVector.reshape(1, vocabSize);
        // Normalize the new email vector
        normalizer.transform(newEmailVector);
        // Get model output
        double[] output = model.output(newEmailVector).toDoubleVector();

        System.out.println("Spam Probability: " + output[0]);
        if (output[0] > 0.5) {
            System.out.println("This is spam!");
        } else {
            System.out.println("This is not spam.");
        }
    }
}