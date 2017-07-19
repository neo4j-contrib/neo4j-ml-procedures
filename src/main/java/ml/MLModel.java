package ml;

import org.encog.ConsoleStatusReportable;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.versatile.NormalizationHelper;
import org.encog.ml.data.versatile.VersatileMLDataSet;
import org.encog.ml.data.versatile.columns.ColumnDefinition;
import org.encog.ml.data.versatile.columns.ColumnType;
import org.encog.ml.data.versatile.sources.VersatileDataSource;
import org.encog.ml.factory.MLMethodFactory;
import org.encog.ml.model.EncogModel;
import org.encog.util.simple.EncogUtility;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author mh
 * @since 19.07.17
 */
public class MLModel {
    private State state;
    private final String name;
    private boolean initialized;
    private EncogModel model; // todo MLMethod and decide later between regression, classification and others
    private MLRegression method;
    private String methodName;

    public void add(Map<String, Object> inputs, Object output) {
        if (this.state == State.created || this.state == State.training) {
            rows.add(asRow(inputs, output));
            this.state = State.training;
        } else {
            throw new IllegalArgumentException(String.format("Model %s not able to accept training data, state is: %s", name, state));
        }
    }

    private String[] asRow(Map<String, Object> inputs, Object output) {
        String[] row = new String[inputs.size() + (output == null ? 0 : 1)];
        for (String k : inputs.keySet()) {
            row[offsets.get(k)] = inputs.get(k).toString();
        }
        if (output != null) {
            row[offsets.get(this.output)] = output.toString();
        }
        return row;
    }

    public void train() {
        if (state != State.ready) {
            if (state != State.training) {
                throw new IllegalArgumentException(String.format("Model %s is not ready to predict, it has no training data, state is %s", name, state));
            }
            initialize();
        }
    }

    public Object predict(Map<String, Object> inputs) {
        if (state != State.ready) {
            train();
        }
        if (state == State.ready) {
            NormalizationHelper helper = model.getDataset().getNormHelper();
            MLData input = helper.allocateInputVector();
            String[] line = asRow(inputs, null);

            helper.normalizeInputVector(line, input.getData(), false);
            MLData output = method.compute(input);
            String predicted = helper.denormalizeOutputVectorToString(output)[0]; // todo multiple outputs, non-string outputs
            // todo confidence
            return predicted;
        } else {
            throw new IllegalArgumentException(String.format("Model %s is not ready to predict, state is %s", name, state));
        }
    }

    private void initialize() {
        VersatileMLDataSet data = new VersatileMLDataSet(new VersatileDataSource() {
            int idx = 0;

            @Override
            public String[] readLine() {
                return idx >= rows.size() ? null : rows.get(idx++);
            }

            @Override
            public void rewind() {
                idx = 0;
            }

            @Override
            public int columnIndex(String s) {
                return offsets.get(s);
            }
        });
        offsets.entrySet().stream().sorted(Comparator.comparingInt(Map.Entry::getValue)).forEach(e -> {
            String k = e.getKey();
            ColumnDefinition col = data.defineSourceColumn(k, offsets.get(k), types.get(k)); // todo has bug, doesn't work like that, cols have to be in index order
            if (k.equals(output)) {
                data.defineOutput(col);
            } else {
                data.defineInput(col);
            }
        });
//        types.forEach((k, v) -> {
//            ColumnDefinition col = data.defineSourceColumn(k, offsets.get(k), v); // todo has bug, doesn't work like that, cols have to be in index order
//            if (k.equals(output)) {
//                data.defineOutput(col);
//            } else {
//                data.defineInput(col);
//            }
//        });
        // Analyze the data, determine the min/max/mean/sd of every column.
        data.analyze();

        // Create feedforward neural network as the model type. MLMethodFactory.TYPE_FEEDFORWARD.
        // You could also other model types, such as:
        // MLMethodFactory.SVM:  Support Vector Machine (SVM)
        // MLMethodFactory.TYPE_RBFNETWORK: RBF Neural Network
        // MLMethodFactor.TYPE_NEAT: NEAT Neural Network
        // MLMethodFactor.TYPE_PNN: Probabilistic Neural Network
        EncogModel model = new EncogModel(data);
        model.selectMethod(data, methodName); // todo from config
        // Send any output to the console.
// model.setReport(new ConsoleStatusReportable());

        // Now normalize the data.  Encog will automatically determine the correct normalization
        // type based on the model you chose in the last step.
        data.normalize();

        // Hold back some data for a final validation.
        // Shuffle the data into a random ordering.
        // Use a seed of 1001 so that we always use the same holdback and will get more consistent results.
        model.holdBackValidation(0.3, true, 1001); // todo from config

        // Choose whatever is the default training type for this model.
        model.selectTrainingType(data);

        // Use a 5-fold cross-validated train.  Return the best method found.
        MLRegression bestMethod = (MLRegression) model.crossvalidate(5, true); // todo from config
        // MLRegression vs. MLClassification

        // Display the training and validation errors.
//        System.out.println("Training error: " + EncogUtility.calculateRegressionError(bestMethod, model.getTrainingDataset()));
//        System.out.println("Validation error: " + EncogUtility.calculateRegressionError(bestMethod, model.getValidationDataset()));

        // Display our normalization parameters.
//        NormalizationHelper helper = data.getNormHelper();
//        System.out.println(helper.toString());

        // Display the final model.
//         System.out.println("Final model: " + bestMethod);
        this.model = model;
        this.method = bestMethod;
        this.state = State.ready;
    }

    public static ML.ModelResult remove(String model) {
        MLModel existing = models.remove(model);
        return new ML.ModelResult(model, existing == null ? State.unknown :  State.removed);
    }

    //
    enum Method {
        ffd, svm, rbf, neat, pnn;
    }
    /*
    "feedforward" "svm", "rbfnetwork", "neat", "pnn"
    */

    /*
    nominal,
ordinal,
continuous,
ignore
     */
    private ColumnType typeOf(String type) {
        switch (type.toUpperCase()) {
            case "CLASS":
                return ColumnType.nominal;
            case "FLOAT":
                return ColumnType.continuous;
            case "ORDER":
                return ColumnType.ordinal;
            default:
                throw new IllegalArgumentException("Unknown type: " + type);
        }
    }

    private final Map<String, ColumnType> types = new HashMap<>();
    private final Map<String, Integer> offsets = new HashMap<>();
    private final String output;
    private final Map<String, Object> config;
    private final List<String[]> rows = new ArrayList<>();

    private static ConcurrentHashMap<String,MLModel> models = new ConcurrentHashMap<>();

    public MLModel(String name, Map<String, String> types, String output, Map<String, Object> config) {
        if (models.containsKey(name)) throw new IllegalArgumentException("Model "+name+" already exists, please remove first");
        if (!types.containsKey(output)) throw new IllegalArgumentException("Outputs not defined: " + output);
        this.name = name;
        this.methodName = MLMethodFactory.TYPE_FEEDFORWARD;
        int i = 0;
        for (Map.Entry<String, String> entry : types.entrySet()) {
            String key = entry.getKey();
            this.types.put(key, typeOf(entry.getValue()));
            if (!key.equals(output)) this.offsets.put(key, i++);
        }
        this.offsets.put(output, i);
        this.output = output;
        this.config = config;
        this.state = State.created;
        models.put(name, this);
    }

    public enum State { created, training, ready, removed, unknown}


    public ML.ModelResult asResult() {
        ML.ModelResult result =
                new ML.ModelResult(this.name, this.state)
                .withInfo("methodName", methodName);

        if (rows.size() > 0) {
            result = result.withInfo("trainingSets",(long)rows.size());
        }
        if (state == State.ready) {
            // todo check how expensive this is
            result = result.withInfo(
                "trainingError", EncogUtility.calculateRegressionError(method, model.getTrainingDataset()),
                "validationError",EncogUtility.calculateRegressionError(method, model.getValidationDataset()),
                "selectedMethod",method.toString(),
                "normalization",model.getDataset().getNormHelper().toString()
            );
        }
        return result;
    }

    public static MLModel from(String name) {
        MLModel model = models.get(name);
        if (model != null) return model;
        throw new IllegalArgumentException("No valid ML-Model " + name);
    }
}
