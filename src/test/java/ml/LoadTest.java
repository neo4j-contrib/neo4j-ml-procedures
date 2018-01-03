package ml;

import org.junit.*;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Transaction;
import org.neo4j.test.TestGraphDatabaseFactory;

/**
 * @author mh
 * @since 26.07.17
 */
public class LoadTest {
    @Test
    @Ignore("While parsing a protocol message, the input ended unexpectedly in the middle of a field.  This could mean either that the input has been truncated or that an embedded message misreported its own length.")
    public void loadTensorFlow() throws Exception {
//        String url = getClass().getResource("/tensorflow_example.pbtxt").toString();
        String url = getClass().getResource("/saved_model.pb").toString();
        System.out.println("url = " + url);
        GraphDatabaseService db = new TestGraphDatabaseFactory().newImpermanentDatabase();
        try (Transaction tx = db.beginTx()) {
            LoadTensorFlow load = new LoadTensorFlow();
            load.loadTensorFlow(url);
            tx.success();
        }
    }
}
