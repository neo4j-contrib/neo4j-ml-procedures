package ml;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.datavec.api.records.reader.impl.collection.ListStringRecordReader;
import org.datavec.api.split.ListStringSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * @author mh
 * @since 23.07.17
 */
public class DL4JMLModel extends MLModel {
    private MultiLayerNetwork model;

    public DL4JMLModel(String name, Map<String, String> types, String output, Map<String, Object> config) {
        super(name, types, output, config);
    }

    @Override
    protected Object doPredict(String[] line) {
        try {
            ListStringSplit input = new ListStringSplit(Collections.singletonList(Arrays.asList(line)));
            ListStringRecordReader rr = new ListStringRecordReader();
            rr.initialize(input);
            DataSetIterator iterator = new RecordReaderDataSetIterator(rr, 1);

            DataSet ds = iterator.next();
            INDArray prediction = model.output(ds.getFeatures());

            DataType outputType = types.get(this.output);
            switch (outputType) {
                case _float : return prediction.getDouble(0);
                case _class: {
                    int numClasses = 2;
                    double max = 0;
                    int maxIndex = -1;
                    for (int i=0;i<numClasses;i++) {
                        if (prediction.getDouble(i) > max) {maxIndex = i; max = prediction.getDouble(i);}
                    }
                    return maxIndex;
//                    return prediction.getInt(0,1); // numberOfClasses
                }
                default: throw new IllegalArgumentException("Output type not yet supported "+outputType);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    protected void doTrain() {
        try {
            int seed = 123;
            double learningRate = 0.01;
            int batchSize = 50;
            int nEpochs = 30;

            int numOutputs = 1;
            int numInputs = types.size() - numOutputs;
            int outputOffset = types.size() - numOutputs; // last column
            int numHiddenNodes = 20;
            int numberOfClasses = 2; // todo from normalizer !!!
            double evalPercent = 0.3;

            List<List<String>> data = rows.parallelStream().map(Arrays::asList).collect(Collectors.toList());
            Collections.shuffle(data);

            ListStringSplit input = new ListStringSplit(data.subList(0, (int) (data.size() * (1.0 - evalPercent))));
            ListStringRecordReader rr = new ListStringRecordReader();
            rr.initialize(input);
            DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, outputOffset, numberOfClasses);

            //Load the test/evaluation data:
            ListStringSplit test = new ListStringSplit(data.subList((int) (data.size() * (1.0 - evalPercent)), data.size()));
            ListStringRecordReader rrTest = new ListStringRecordReader();
            rrTest.initialize(test);
            DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, outputOffset, numberOfClasses);

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .iterations(1)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .learningRate(learningRate)
                    .updater(Updater.NESTEROVS).momentum(0.9)
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                            .weightInit(WeightInit.XAVIER)
                            .activation(Activation.RELU)
                            .build())
                    .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                            .weightInit(WeightInit.XAVIER)
                            .activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER)
                            .nIn(numHiddenNodes).nOut(numberOfClasses).build())
                    .pretrain(false).backprop(true).build();


            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            model.setListeners(new ScoreIterationListener(10));  //Print score every 10 parameter updates


            for (int n = 0; n < nEpochs; n++) {
                model.fit(trainIter);
            }

            System.out.println("Evaluate model....");
            Evaluation eval = new Evaluation(numOutputs);
            while (testIter.hasNext()) {
                DataSet t = testIter.next();
                INDArray features = t.getFeatureMatrix();
                INDArray labels = t.getLabels();
                INDArray predicted = model.output(features, false);

                eval.eval(labels, predicted);

            }

            //Print the evaluation statistics
            System.out.println(eval.stats());

            this.model = model;
            this.state = State.ready;

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
