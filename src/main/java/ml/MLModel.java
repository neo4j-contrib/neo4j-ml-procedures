package ml;

import org.encog.ml.factory.MLMethodFactory;
import org.neo4j.graphdb.Node;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author mh
 * @since 23.07.17
 */
public abstract class MLModel<ROW> {

    static class Config {
        private final Map<String, Object> config;

        public Config(Map<String, Object> config) {
            this.config = config;
        }

        class V<T> {
            private final String name;
            private final T defaultValue;

            V(String name, T defaultValue) {
                this.name = name;
                this.defaultValue = defaultValue;
            }

            T get(T defaultValue) {
                Object value = config.get(name);
                if (value == null) return defaultValue;
                if (defaultValue instanceof Double) return (T) (Object) ((Number) value).doubleValue();
                if (defaultValue instanceof Integer) return (T) (Object) ((Number) value).intValue();
                if (defaultValue instanceof Long) return (T) (Object) ((Number) value).longValue();
                if (defaultValue instanceof String) return (T) value.toString();
                return (T) value;
            }

            T get() {
                return get(defaultValue);
            }
        }

        public final V<Long> seed = new V<>("seed", 123L);
        public final V<Double> learningRate = new V<>("learningRate", 0.01d);
        public final V<Integer> epochs = new V<>("epochs", 50);
        public final V<Integer> hidden = new V<>("hidden", 20);
        public final V<Double> trainPercent = new V<>("trainPercent", 0.75d);
    }

    static ConcurrentHashMap<String, MLModel> models = new ConcurrentHashMap<>();
    final String name;
    final Map<String, DataType> types = new HashMap<>();
    final Map<String, Integer> offsets = new HashMap<>();
    final String output;
    final Config config;
    final List<ROW> rows = new ArrayList<>();
    State state;
    Method methodName;

    public MLModel(String name, Map<String, String> types, String output, Map<String, Object> config) {
        if (models.containsKey(name))
            throw new IllegalArgumentException("Model " + name + " already exists, please remove first");

        this.name = name;
        this.state = State.created;
        this.output = output;
        this.config = new Config(config);
        initTypes(types, output);

        this.methodName = Method.ffd;

        models.put(name, this);

    }

    protected void initTypes(Map<String, String> types, String output) {
        if (!types.containsKey(output)) throw new IllegalArgumentException("Outputs not defined: " + output);
        int i = 0;
        for (Map.Entry<String, String> entry : types.entrySet()) {
            String key = entry.getKey();
            this.types.put(key, DataType.from(entry.getValue()));
            if (!key.equals(output)) this.offsets.put(key, i++);
        }
        this.offsets.put(output, i);
    }

    public static ML.ModelResult remove(String model) {
        MLModel existing = models.remove(model);
        return new ML.ModelResult(model, existing == null ? State.unknown : State.removed);
    }

    public static MLModel<?> from(String name) {
        MLModel model = models.get(name);
        if (model != null) return model;
        throw new IllegalArgumentException("No valid ML-Model " + name);
    }

    public void add(Map<String, Object> inputs, Object output) {
        if (this.state == State.created || this.state == State.training) {
            rows.add(asRow(inputs, output));
            this.state = State.training;
        } else {
            throw new IllegalArgumentException(String.format("Model %s not able to accept training data, state is: %s", name, state));
        }
    }

    protected abstract ROW asRow(Map<String, Object> inputs, Object output);

    public void train() {
        if (state != State.ready) {
            if (state != State.training) {
                throw new IllegalArgumentException(String.format("Model %s is not ready to predict, it has no training data, state is %s", name, state));
            }
            doTrain();
        }
    }

    public Object predict(Map<String, Object> inputs) {
        if (state != State.ready) {
            train();
        }
        if (state == State.ready) {
            ROW line = asRow(inputs, null);

            Object predicted = doPredict(line);
            // todo confidence
            return predicted;
        } else {
            throw new IllegalArgumentException(String.format("Model %s is not ready to predict, state is %s", name, state));
        }
    }

    protected abstract Object doPredict(ROW line);

    protected abstract void doTrain();

    public ML.ModelResult asResult() {
        ML.ModelResult result =
                new ML.ModelResult(this.name, this.state)
                        .withInfo("methodName", methodName);

        if (rows.size() > 0) {
            result = result.withInfo("trainingSets", (long) rows.size());
        }
        if (state == State.ready) {
            // todo check how expensive this is
            result = resultWithInfo(result);
        }
        return result;
    }

    protected ML.ModelResult resultWithInfo(ML.ModelResult result) {
        return result;
    }

    ;

    public static MLModel create(String name, Map<String, String> types, String output, Map<String, Object> config) {
        String framework = config.getOrDefault("framework", "encog").toString().toLowerCase();
        switch (framework) {
            case "encog":
                return new EncogMLModel(name, types, output, config);
            case "dl4j":
                return new DL4JMLModel(name, types, output, config);
            default:
                throw new IllegalArgumentException("Unknown framework: " + framework);
        }
    }

    enum Method {
        ffd, svm, rbf, neat, pnn;
    }

    enum DataType {
        _class, _float, _order;

        public static DataType from(String type) {
            switch (type.toUpperCase()) {
                case "CLASS":
                    return DataType._class;
                case "FLOAT":
                    return DataType._float;
                case "ORDER":
                    return DataType._order;
                default:
                    throw new IllegalArgumentException("Unknown type: " + type);
            }
        }
    }

    public enum State {created, training, ready, removed, unknown}

    List<Node> show() {
        return Collections.emptyList();
    }
}
