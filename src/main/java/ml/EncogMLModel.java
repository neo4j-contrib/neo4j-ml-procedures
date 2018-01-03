package ml;

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
import org.neo4j.graphdb.Node;

import java.util.*;

/**
 * @author mh
 * @since 19.07.17
 */
public class EncogMLModel extends MLModel<String[]> {

    private EncogModel model; // todo MLMethod and decide later between regression, classification and others
    private MLRegression method;


    @Override
    protected String[] asRow(Map<String, Object> inputs, Object output)  {
        String[] row = new String[inputs.size() + (output == null ? 0 : 1)];
        for (String k : inputs.keySet()) {
            row[offsets.get(k)] = inputs.get(k).toString();
        }
        if (output != null) {
            row[offsets.get(this.output)] = output.toString();
        }
        return row;
    }


    @Override
    protected Object doPredict(String[] line) {
        NormalizationHelper helper = model.getDataset().getNormHelper();
        MLData input = helper.allocateInputVector();
        helper.normalizeInputVector(line, input.getData(), false);
        MLData output = method.compute(input);
        DataType outputType = types.get(this.output);
        switch (outputType) {
            case _float : return output.getData(0);
            case _class: return helper.denormalizeOutputVectorToString(output)[0];
            default: throw new IllegalArgumentException("Output type not yet supported "+outputType);
        }
    }

    @Override
    protected void doTrain() {
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
            ColumnDefinition col = data.defineSourceColumn(k, offsets.get(k), typeOf(types.get(k))); // todo has bug, doesn't work like that, cols have to be in index order
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
        model.selectMethod(data, methodFor(methodName)); // todo from config
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

    private String methodFor(Method method) {
        switch (method) {
            case ffd: return  MLMethodFactory.TYPE_FEEDFORWARD;
            case svm: return MLMethodFactory.TYPE_SVM;
            case rbf: return MLMethodFactory.TYPE_RBFNETWORK;
            case neat: return MLMethodFactory.TYPE_NEAT;
            case pnn: return MLMethodFactory.TYPE_PNN;
            default: throw new IllegalArgumentException("Unknown method "+method);
        }
    }

    /*
    nominal,
ordinal,
continuous,
ignore
     */
    private ColumnType typeOf(DataType type) {
        switch (type) {
            case _class:
                return ColumnType.nominal;
            case _float:
                return ColumnType.continuous;
            case _order:
                return ColumnType.ordinal;
            default:
                throw new IllegalArgumentException("Unknown type: " + type);
        }
    }

    public EncogMLModel(String name, Map<String, String> types, String output, Map<String, Object> config) {
        super(name, types, output, config);
    }


    @Override
    protected ML.ModelResult resultWithInfo(ML.ModelResult result) {
        return result.withInfo(
            "trainingError", EncogUtility.calculateRegressionError(method, model.getTrainingDataset()),
            "validationError",EncogUtility.calculateRegressionError(method, model.getValidationDataset()),
            "selectedMethod",method.toString(),
            "normalization",model.getDataset().getNormHelper().toString()
        );
    }
}
