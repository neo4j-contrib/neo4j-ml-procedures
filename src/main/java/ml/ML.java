package ml;

import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.logging.Log;
import org.neo4j.procedure.Context;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.Procedure;

import java.util.*;
import java.util.stream.Stream;

public class ML {
    @Context
    public GraphDatabaseService db;

    @Context
    public Log log;

    /*

    apoc.ml.create.classifier({types},['output'],{config}) yield model
apoc.ml.create.regression({params},{config}) yield model

apoc.ml.train(model, {params}, prediction)

apoc.ml.classify(model, {params}) yield prediction:string, confidence:float
apoc.ml.regression(model, {params}) yield prediction:float, confidence:float

apoc.ml.delete(model) yield model

     */

    @Procedure
    public Stream<ModelResult> create(@Name("model") String model, @Name("types") Map<String,String> types, @Name(value="output") String output, @Name(value="params",defaultValue="{}") Map<String, Object> config) {
        return Stream.of(MLModel.create(model,types,output,config).asResult());
    }

    @Procedure
    public Stream<ModelResult> info(@Name("model") String model) {
        return Stream.of(MLModel.from(model).asResult());
    }

    @Procedure
    public Stream<ModelResult> remove(@Name("model") String model) {
        return Stream.of(MLModel.remove(model));
    }

    @Procedure
    public Stream<ModelResult> add(@Name("model") String model, @Name("inputs") Map<String,Object> inputs, @Name("outputs") Object output) {
        MLModel mlModel = MLModel.from(model);
        mlModel.add(inputs,output);
        return Stream.of(mlModel.asResult());
    }
    @Procedure
    public Stream<ModelResult> train(@Name("model") String model) {
        MLModel mlModel = MLModel.from(model);
        mlModel.train();
        return Stream.of(mlModel.asResult());
    }
    @Procedure
    public Stream<PredictionResult> predict(@Name("model") String model, @Name("inputs") Map<String,Object> inputs) {
        MLModel mlModel = MLModel.from(model);
        Object value = mlModel.predict(inputs);
        double confidence = 0.0d;
        return Stream.of(new PredictionResult(value, confidence));
    }

    public static class PredictionResult {
        public Object value;
        public double confidence;

        public PredictionResult(Object value, double confidence) {
            this.value = value;
            this.confidence = confidence;
        }
    }

    public static class ModelResult {
        public final String model;
        public final String state;
        public final Map<String,Object> info = new HashMap<>();

        public ModelResult(String model, EncogMLModel.State state) {
            this.model = model;
            this.state = state.name();
        }

        ModelResult withInfo(Object...infos) {
            for (int i = 0; i < infos.length; i+=2) {
                info.put(infos[i].toString(),infos[i+1]);
            }
            return this;
        }
    }

}
