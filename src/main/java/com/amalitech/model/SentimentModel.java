package com.amalitech.model;

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

public class SentimentModel {
    private MultiLayerNetwork model;
    private DataNormalization normalizer;
    private int vocabSize;
    private DefaultTokenizerFactory tokenizerFactory;

    public SentimentModel(MultiLayerNetwork model, DataNormalization normalizer, int vocabSize, DefaultTokenizerFactory tokenizerFactory) {
        this.model = model;
        this.normalizer = normalizer;
        this.vocabSize = vocabSize;
        this.tokenizerFactory = tokenizerFactory;
    }

    /**
     * Analyzes sentiment for a given text input.
     * @param text The input text.
     * @return A probability (between 0 and 1); above 0.5 is considered positive.
     */
    public double analyzeSentiment(String text) {
        INDArray inputVector = vectorizeText(text);
        normalizer.transform(inputVector);
        double[] output = model.output(inputVector).toDoubleVector();
        return output[0];
    }

    /**
     * Vectorizes a text string into an INDArray of shape [1, vocabSize].
     */
    private INDArray vectorizeText(String text) {
        INDArray textVector = Nd4j.create(vocabSize);
        tokenizerFactory.create(text).getTokens().forEach(token -> {
            int index = Math.abs(token.hashCode() % vocabSize);
            textVector.putScalar(index, textVector.getDouble(index) + 1);
        });
        return textVector.reshape(1, vocabSize);
    }

    /**
     * Saves the model (and normalizer) to the specified file.
     */
    public void saveModel(File file) throws Exception {
        ModelSerializer.writeModel(model, file, true, normalizer);
    }

    /**
     * Loads a saved model (and normalizer) from file if available; otherwise trains a new model.
     */
    public static SentimentModel loadOrTrain(File modelFile, File trainingDataFile, int vocabSize) throws Exception {
        DefaultTokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        MultiLayerNetwork model;
        DataNormalization normalizer;

        if (modelFile.exists()) {
            System.out.println("Loading saved model...");
            model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
            normalizer = ModelSerializer.restoreNormalizerFromFile(modelFile);
        } else {
            System.out.println("No saved model found. Training a new model...");

            File modelToSave = new File("src/main/resources/sentiments.zip");
            DataSet trainingData = loadTrainingData(modelToSave, vocabSize, tokenizerFactory);

            // Normalize the dataset
            normalizer = new NormalizerStandardize();
            normalizer.fit(trainingData);
            normalizer.transform(trainingData);

            // Create and train the model
            MultiLayerNetwork newModel = new MultiLayerNetwork(ModelConfiguration.createMultiLayerConfiguration(vocabSize));
            newModel.init();
            newModel.setListeners(new ScoreIterationListener(10));
            newModel.fit(trainingData);
            model = newModel;

            // Save the model along with the normalizer
            ModelSerializer.writeModel(model, modelFile, true, normalizer);
            System.out.println("Model trained and saved to " + modelFile.getAbsolutePath());
        }
        return new SentimentModel(model, normalizer, vocabSize, tokenizerFactory);
    }

    /**
     * Loads training data from a CSV file where each line is formatted as:
     * label, text
     */
    private static DataSet loadTrainingData(File csvFile, int vocabSize, DefaultTokenizerFactory tokenizerFactory) throws Exception {
        List<String> texts = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        BasicLineIterator iterator = new BasicLineIterator(csvFile);
        while (iterator.hasNext()) {
            String line = iterator.nextSentence();
            String[] parts = line.split(",", 2);
            labels.add(Integer.parseInt(parts[0].trim()));
            texts.add(parts[1].trim());
        }
        int numExamples = texts.size();
        INDArray features = Nd4j.create(numExamples, vocabSize);
        INDArray labelsArray = Nd4j.create(numExamples, 1);

        for (int i = 0; i < numExamples; i++) {
            String text = texts.get(i);
            INDArray textVector = Nd4j.create(vocabSize);
            tokenizerFactory.create(text).getTokens().forEach(token -> {
                int index = Math.abs(token.hashCode() % vocabSize);
                textVector.putScalar(index, textVector.getDouble(index) + 1);
            });
            features.putRow(i, textVector);
            labelsArray.putScalar(i, labels.get(i));
        }
        return new DataSet(features, labelsArray);
    }
}

