package ml;

import org.datavec.api.records.metadata.RecordMetaData;
import org.deeplearning4j.datasets.iterator.DoublesDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.meta.Prediction;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static org.nd4j.linalg.primitives.Pair.makePair;


/**
 * @author mh
 * @since 18.07.17
 */
public class DL4JXORHelloWorld {

    /**
     * The input necessary for XOR.
     */
    public static double XOR_INPUT[][] = { { 0.0, 0.0 }, { 1.0, 0.0 },
            { 0.0, 1.0 }, { 1.0, 1.0 } };

    /**
     * The ideal data necessary for XOR.
     */
    public static double XOR_IDEAL[][] = { { 0.0 }, { 1.0 }, { 1.0 }, { 0.0 } };

    /**
     * The main method.
     * @param args No arguments are used.
     */
    public static void main(final String args[]) throws IOException, InterruptedException {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.1)
//                 .useDropConnect(false)
//                .biasInit(0)
                .miniBatch(false)
//                .updater(Updater.SGD)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2)
//                        .weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(0,1))
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
//                        .weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(0,1))
                        .activation(Activation.SIGMOID)
                        .nIn(2).nOut(1).build())
                .pretrain(false).backprop(true).build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(1));  //Print score every 10 parameter updates


        DoublesDataSetIterator iterator = new DoublesDataSetIterator(Arrays.asList(
                makePair(XOR_INPUT[0],XOR_IDEAL[0]),
                makePair(XOR_INPUT[1],XOR_IDEAL[1]),
                makePair(XOR_INPUT[2],XOR_IDEAL[2]),
                makePair(XOR_INPUT[3],XOR_IDEAL[3]))
                ,1);

        for (int n = 0; n < 10000; n++) {
            model.fit(iterator);
        }

        Evaluation eval = model.evaluate(iterator);
        List<Prediction> predictionErrors = eval.getPredictionErrors();
        System.out.println("\n\n+++++ Prediction Errors +++++");
        if (predictionErrors != null) {
            for (Prediction p : predictionErrors) {
                System.out.printf("Predicted class: %d, Actual class: %d\t%s%n", p.getPredictedClass(), p.getActualClass(), p.getRecordMetaData(RecordMetaData.class));
            }
        }
        //Print the evaluation statistics
        System.out.println(eval.stats());

        INDArray data = Nd4j.zeros(2, 2);
        data.putScalar(0,0,0d);
        data.putScalar(0,1,1d);
        data.putScalar(1,0,1d);
        data.putScalar(1,1,1d);
        INDArray output = model.output(data);

        for (int i=0;i<data.rows();i++) {
            System.out.println(data.getRow(i) +" -> "+ output.getRow(i));
        }
    }

    {
        // ModelSerializer.writeModel(model, stream, true);
        // works also with hdfds
        // ModelSerializer.restoreMultiLayerNetwork(fileOrStream)

    }
}

