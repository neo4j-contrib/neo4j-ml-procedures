package ml;

import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Result;
import org.neo4j.kernel.impl.proc.Procedures;
import org.neo4j.kernel.internal.GraphDatabaseAPI;
import org.neo4j.test.TestGraphDatabaseFactory;

import java.io.File;
import java.util.Collections;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.neo4j.helpers.collection.MapUtil.map;
import static org.neo4j.helpers.collection.MapUtil.stringMap;

/**
 * @author mh
 * @since 19.07.17
 */
public class MLProcedureTest {

    private GraphDatabaseService db;

    @Before
    public void setUp() throws Exception {
        db = new TestGraphDatabaseFactory().newImpermanentDatabase();
        Procedures procedures = ((GraphDatabaseAPI) db).getDependencyResolver().resolveDependency(Procedures.class);
        procedures.registerProcedure(ML.class);
    }

    @After
    public void tearDown() throws Exception {
        db.shutdown();
    }

    @Test
    public void predict() throws Exception {
        Map<String, String> types = stringMap("sepal-length", "float", "sepal-width", "float", "petal-length", "float", "petal-width", "float", "iris", "class");
        Result result = db.execute("CALL ml.create({model},{types},{output})", map("model", "iris", "types", types, "output", "iris"));
        System.out.println("result.resultAsString() = " + result.resultAsString());

        File irisFile = new File(getClass().getResource("/iris.csv").getFile());
        ReadCSV csv = new ReadCSV(irisFile, false, CSVFormat.DECIMAL_POINT);
        while (csv.next()) {
            Map<String, Object> inputs = map("sepal-length", csv.get(0), "sepal-width", csv.get(1), "petal-length", csv.get(2), "petal-width", csv.get(3));
            Map<String, Object> params = map("model","iris","inputs",inputs,"output",csv.get(4));
            db.execute("CALL ml.add({model},{inputs},{output})",params).close();
        }
        csv.close();
        System.out.println(db.execute("CALL ml.train('iris')").resultAsString());
        csv = new ReadCSV(irisFile, false, CSVFormat.DECIMAL_POINT);
        int total = 0, correct = 0;
        while (csv.next()) {
            Map<String, Object> inputs = map("sepal-length", csv.get(0), "sepal-width", csv.get(1), "petal-length", csv.get(2), "petal-width", csv.get(3));
            Map<String, Object> params = map("model","iris","inputs",inputs);
            Object predicted = db.execute("CALL ml.predict({model},{inputs})",params).columnAs("value").next();
            total++;
            if (csv.get(4).equals(predicted)) {
                correct++;
            }
        }
        csv.close();
        System.out.printf("Predictions correct %d of %d%n",correct,total);
        System.out.println(db.execute("CALL ml.info('iris')").resultAsString());
        System.out.println(db.execute("CALL ml.remove('iris')").resultAsString());
        assertEquals(total,correct,3d);

    }

}
