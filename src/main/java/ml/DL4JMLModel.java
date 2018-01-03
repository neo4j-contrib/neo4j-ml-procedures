package ml;

import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.impl.collection.ListStringRecordReader;
import org.datavec.api.split.ListStringSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.meta.Prediction;
import org.deeplearning4j.nn.api.Layer;
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
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.cpu.nativecpu.CpuNDArrayFactory;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.util.NDArrayUtil;
import org.neo4j.graphdb.Label;
import org.neo4j.graphdb.Node;
import org.neo4j.helpers.collection.MapUtil;
import result.VirtualNode;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * @author mh
 * @since 23.07.17
 */
public class DL4JMLModel extends MLModel<List<String>> {
    private MultiLayerNetwork model;

    public DL4JMLModel(String name, Map<String, String> types, String output, Map<String, Object> config) {
        super(name, types, output, config);
    }

    @Override
    protected List<String> asRow(Map<String, Object> inputs, Object output) {
        List<String> row = new ArrayList<>(inputs.size() + (output == null ? 0 : 1));
        for (String k : inputs.keySet()) {
            row.add(offsets.get(k), inputs.get(k).toString());
        }
        if (output != null) {
            row.add(offsets.get(this.output), output.toString());
        }
        return row;
    }

    @Override
    protected Object doPredict(List<String> line) {
        try {
            ListStringSplit input = new ListStringSplit(Collections.singletonList(line));
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
            long seed = config.seed.get();
            double learningRate = config.learningRate.get();
            int nEpochs = config.epochs.get();

            int numOutputs = 1;
            int numInputs = types.size() - numOutputs;
            int outputOffset = offsets.get(output); // last column
            int numHiddenNodes = config.hidden.get();
            double trainPercent = config.trainPercent.get();
            int batchSize = rows.size(); // full dataset size

            Map<String,Set<String>> classes = new HashMap<>();
            types.entrySet().stream()
                    .filter(e -> e.getValue() == DataType._class)
                    .map(e -> new HashMap.SimpleEntry<>(e.getKey(), offsets.get(e.getKey())))
                    .forEach(e -> classes.put(e.getKey(),rows.parallelStream().map(r -> r.get(e.getValue())).distinct().collect(Collectors.toSet())));

            int numberOfClasses = (int)classes.get("output").size();
            System.out.println("labels = " + classes);

            ListStringSplit input = new ListStringSplit(rows);
            ListStringRecordReader rr = new ListStringRecordReader();
            rr.initialize(input);
            RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(rr, batchSize, outputOffset, numberOfClasses);

            iterator.setCollectMetaData(true);  // Instruct the iterator to collect metadata, and store it in the DataSet objects
            DataSet allData = iterator.next();
            allData.shuffle(seed);
            SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(trainPercent);  //Use 65% of data for training

            DataSet trainingData = testAndTrain.getTrain();
            DataSet testData = testAndTrain.getTest();

            //Normalize data as per basic CSV example
//            NormalizerStandardize normalizer = new NormalizerStandardize();
            NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler();
            normalizer.fitLabel(true);
            normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
            normalizer.transform(trainingData);     //Apply normalization to the training data
            normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set

            //Let's view the example metadata in the training and test sets:
            List<RecordMetaData> trainMetaData = trainingData.getExampleMetaData(RecordMetaData.class);
            List<RecordMetaData> testMetaData = testData.getExampleMetaData(RecordMetaData.class);

            //Let's show specifically which examples are in the training and test sets, using the collected metadata
//            System.out.println("  +++++ Training Set Examples MetaData +++++");
//            String format = "%-20s\t%s";
//            for(RecordMetaData recordMetaData : trainMetaData){
//                System.out.println(String.format(format, recordMetaData.getLocation(), recordMetaData.getURI()));
//                //Also available: recordMetaData.getReaderClass()
//            }
//            System.out.println("\n\n  +++++ Test Set Examples MetaData +++++");
//            for(RecordMetaData recordMetaData : testMetaData){
//                System.out.println(recordMetaData.getLocation());
//            }



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
                model.fit(trainingData);
            }

            System.out.println("Evaluate model....");
            INDArray output = model.output(testData.getFeatureMatrix(),false);
            Evaluation eval = new Evaluation(numberOfClasses);
            eval.eval(testData.getLabels(), output, testMetaData);          //Note we are passing in the test set metadata here

            List<Prediction> predictionErrors = eval.getPredictionErrors();
            System.out.println("\n\n+++++ Prediction Errors +++++");
            for(Prediction p : predictionErrors){
                System.out.printf("Predicted class: %d, Actual class: %d\t%s%n", p.getPredictedClass(), p.getActualClass(), p.getRecordMetaData(RecordMetaData.class));
            }
            //Print the evaluation statistics
            System.out.println(eval.stats());

            this.model = model;
            this.state = State.ready;

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    List<Node> show() {
        if ( state != State.ready ) throw new IllegalStateException("Model not trained yet");
        List<Node> result = new ArrayList<>();
        int layerCount = model.getnLayers();
        for (Layer layer : model.getLayers()) {
            Node node = node("Layer",
                    "type", layer.type().name(), "index", layer.getIndex(),
                    "pretrainLayer", layer.isPretrainLayer(), "miniBatchSize", layer.getInputMiniBatchSize(),
                    "numParams", layer.numParams());
            if (layer instanceof DenseLayer) {
                DenseLayer dl = (DenseLayer) layer;
                node.addLabel(Label.label("DenseLayer"));
                node.setProperty("activation",dl.getActivationFn().toString()); // todo parameters
                node.setProperty("biasInit",dl.getBiasInit());
                node.setProperty("biasLearningRate",dl.getBiasLearningRate());
                node.setProperty("l1",dl.getL1());
                node.setProperty("l1Bias",dl.getL1Bias());
                node.setProperty("l2",dl.getL2());
                node.setProperty("l2Bias",dl.getL2Bias());
                node.setProperty("distribution",dl.getDist().toString());
                node.setProperty("in",dl.getNIn());
                node.setProperty("out",dl.getNOut());
            }
            result.add(node);
//            layer.preOutput(allOne, Layer.TrainingMode.TEST);
//            layer.p(allOne, Layer.TrainingMode.TEST);
//            layer.activate(allOne, Layer.TrainingMode.TEST);
        }
        return result;
    }

    private Node node(String label, Object...keyValues) {
        return new VirtualNode(new Label[] {Label.label(label)}, MapUtil.map(keyValues),null);
    }
}
