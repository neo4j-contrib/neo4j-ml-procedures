package ml;

import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;
import org.junit.Test;
import org.neo4j.helpers.collection.MapUtil;

import java.io.File;
import java.util.Collections;
import java.util.Map;
import java.util.stream.Stream;

import static org.junit.Assert.*;
import static org.neo4j.helpers.collection.MapUtil.map;
import static org.neo4j.helpers.collection.MapUtil.stringMap;

/**
 * @author mh
 * @since 19.07.17
 */
public class MLTest {
    @Test
    public void predict() throws Exception {
        ML ml = new ML();
        Map<String, String> types = stringMap("sepal-length", "float", "sepal-width", "float", "petal-length", "float", "petal-width", "float", "iris", "class");
        String model = ml.create("iris",types, "iris", Collections.emptyMap()).findAny().get().model;

        File irisFile = new File(getClass().getResource("/iris.csv").getFile());
        ReadCSV csv = new ReadCSV(irisFile, false, CSVFormat.DECIMAL_POINT);
        while (csv.next()) {
            Map<String, Object> inputs = map("sepal-length", csv.get(0), "sepal-width", csv.get(1), "petal-length", csv.get(2), "petal-width", csv.get(3));
            ml.add(model, inputs, csv.get(4));
        }
        csv.close();

        csv = new ReadCSV(irisFile, false, CSVFormat.DECIMAL_POINT);
        int total = 0, correct = 0;
        while (csv.next()) {
            Map<String, Object> inputs = map("sepal-length", csv.get(0), "sepal-width", csv.get(1), "petal-length", csv.get(2), "petal-width", csv.get(3));
            Object predicted = ml.predict(model, inputs).findAny().get().value;
            total++;
            if (csv.get(4).equals(predicted)) {
                correct++;
            }
        }
        csv.close();
        assertEquals(total,correct,3d);
    }

}
