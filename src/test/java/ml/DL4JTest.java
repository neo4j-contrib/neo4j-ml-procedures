package ml;

import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;
import org.junit.Test;

import java.io.File;
import java.net.URL;
import java.util.Collections;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.neo4j.helpers.collection.MapUtil.map;
import static org.neo4j.helpers.collection.MapUtil.stringMap;

/**
 * @author mh
 * @since 19.07.17
 */
public class DL4JTest {
    @Test
    public void predict() throws Exception {
        ML ml = new ML();
        Map<String, String> types = stringMap("val1", "float", "val2", "float", "output", "class");
        String model = ml.create("cl-lin",types, "output", Collections.singletonMap("framework","dl4j")).findAny().get().model;

        URL trainData = new URL("https://raw.githubusercontent.com/deeplearning4j/dl4j-examples/master/dl4j-examples/src/main/resources/classification/linear_data_train.csv");
        ReadCSV csv = new ReadCSV(trainData.openStream(), false, CSVFormat.DECIMAL_POINT);
        while (csv.next()) {
            Map<String, Object> inputs = map("val1", csv.get(1), "val2", csv.get(2));
            ml.add(model, inputs, csv.get(0));
        }
        csv.close();

        URL evalData = new URL("https://raw.githubusercontent.com/deeplearning4j/dl4j-examples/master/dl4j-examples/src/main/resources/classification/linear_data_eval.csv");
        csv = new ReadCSV(evalData.openStream(), false, CSVFormat.DECIMAL_POINT);
        int total = 0, correct = 0;
        while (csv.next()) {
            Map<String, Object> inputs = map("val1", csv.get(1), "val2", csv.get(2));
            Object predicted = ml.predict(model, inputs).findAny().get().value;
            total++;
            if (csv.get(0).equals(predicted.toString())) {
                correct++;
            }
        }
        csv.close();
        assertEquals(total,correct,3d);
    }

}
