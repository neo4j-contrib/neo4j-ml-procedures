package regression;

import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.bytedeco.javacv.FrameFilter;
import org.neo4j.graphdb.Entity;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.ResourceIterator;
import org.neo4j.logging.Log;
import org.neo4j.procedure.*;
import org.neo4j.procedure.Mode;
import org.neo4j.unsafe.impl.batchimport.cache.ByteArray;
import sun.java2d.pipe.SpanShapeRenderer;

import java.io.*;
import java.util.*;
import java.util.stream.Stream;

public class LR {
    @Context
    public GraphDatabaseService db;

    @Context
    public Log log;

    @Procedure(value = "regression.linear.create", mode = Mode.READ)
    public Stream<ModelResult> create(@Name("model") String model) {
        return Stream.of((new LRModel(model)).asResult());
    }

    @Procedure(value = "regression.linear.info", mode = Mode.READ)
    public Stream<ModelResult> info(@Name("model") String model) {
        LRModel lrModel = LRModel.from(model);
        return Stream.of(lrModel.asResult());
    }

    @Procedure(value = "regression.linear.add", mode = Mode.READ)
    public void add(@Name("model") String model, @Name("given") double given, @Name("expected") double expected) {
        LRModel lrModel = LRModel.from(model);
        lrModel.add(given, expected);
    }

    @Procedure(value = "regression.linear.remove", mode = Mode.READ)
    public void remove(@Name("model") String model, @Name("given") double given, @Name("expected") double expected) {
        LRModel lrModel = LRModel.from(model);
        lrModel.removeData(given, expected);
    }

    @Procedure(value = "regression.linear.delete", mode = Mode.READ)
    public Stream<ModelResult> delete(@Name("model") String model) {
        return Stream.of(LRModel.removeModel(model));
    }

    @UserFunction(value = "regression.linear.predict")
    public double predict(@Name("mode") String model, @Name("given") double given) {
        LRModel lrModel = LRModel.from(model);
        return lrModel.predict(given);
    }

    @UserFunction(value = "regression.linear.serialize")
    public Object serialize(@Name("model") String model) {
        LRModel lrModel = LRModel.from(model);
        return lrModel.serialize();
    }

    @Procedure(value = "regression.linear.load", mode = Mode.READ)
    public Stream<ModelResult> load(@Name("model") String model, @Name("data") Object data) {
        SimpleRegression R;
        try { R = (SimpleRegression) convertFromBytes((byte[]) data); }
        catch (Exception e) {
            throw new RuntimeException("invalid data");
        }
        return Stream.of((new LRModel(model, R)).asResult());
    }

    public static class ModelResult {
        public final String model;
        public final String state;
        public final double N;
        public final Map<String,Object> info = new HashMap<>();

        public ModelResult(String model, LRModel.State state, double N) {
            this.model = model;
            this.state = state.name();
            this.N = N;
        }

        ModelResult withInfo(Object...infos) {
            for (int i = 0; i < infos.length; i+=2) {
                info.put(infos[i].toString(),infos[i+1]);
            }
            return this;
        }
    }

    //Serializes the object into a byte array for storage
    public static byte[] convertToBytes(Object object) throws IOException {
        try (ByteArrayOutputStream bos = new ByteArrayOutputStream();
             ObjectOutput out = new ObjectOutputStream(bos)) {
            out.writeObject(object);
            return bos.toByteArray();
        }
    }

    //de serializes the byte array and returns the stored object
    private static Object convertFromBytes(byte[] bytes) throws IOException, ClassNotFoundException {
        try (ByteArrayInputStream bis = new ByteArrayInputStream(bytes);
             ObjectInput in = new ObjectInputStream(bis)) {
            return in.readObject();
        }
    }




}