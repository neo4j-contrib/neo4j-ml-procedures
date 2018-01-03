package ml;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.neo4j.graphdb.*;
import org.neo4j.kernel.impl.proc.Procedures;
import org.neo4j.kernel.internal.GraphDatabaseAPI;
import org.neo4j.test.TestGraphDatabaseFactory;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

import static ml.DeepwalkGraphTest.TestData.Labels.LEAF;
import static ml.DeepwalkGraphTest.TestData.Labels.NODE;
import static ml.DeepwalkGraphTest.TestData.Rels.NEXT;

/**
 * @author p.meltzer@braintree.com
 */
public class DeepwalkGraphTest {

    private GraphDatabaseService db;

    @Before
    public void setUp() throws Exception {
        db = new TestGraphDatabaseFactory().newImpermanentDatabase();
        Procedures procedures = ((GraphDatabaseAPI) db).getDependencyResolver().resolveDependency(Procedures.class);
        procedures.registerProcedure(Deepwalk.class);
    }

    @After
    public void tearDown() throws Exception {
        db.shutdown();
    }

    @Test
    public void vectoriseTestData() throws Exception {

        System.out.println("Populating database with test data.");
        TestData testData = new TestData(db);
        testData.createTreeDataSet(5);

        System.out.println("Fitting deepwalk model and saving vectors to db.");
        Deepwalk.vectoriseAllNodes(db,
                0.01,
                123,
                2,
                10,
                40,
                "deepwalkVector");

        System.out.println("Getting vectors for each node.");
        try (Transaction tx = db.beginTx()) {
            db.getAllNodes().stream()
                    .map(node -> Arrays.toString((double[]) node.getProperty("deepwalkVector")))
                    .forEach(System.out::println);
            tx.success();
        }

    }

    public static class TestData {
    
        private GraphDatabaseService db;

        enum Labels implements Label {
            NODE, LEAF
        }
    
        enum Rels implements RelationshipType {
            NEXT
        }
    
        public TestData(GraphDatabaseService db) {
            this.db = db;
        }

        public void createTreeDataSet(int layers) {
    
            try (Transaction tx = db.beginTx()) {
    
                AtomicInteger currentLayer = new AtomicInteger(0);
                AtomicInteger degree = new AtomicInteger(1);
    
                // create root node
                Node root = db.createNode(NODE, LEAF);
                root.setProperty("layer", currentLayer.get());
                root.setProperty("degree", degree.getAndIncrement());
    
                // general case
                while (currentLayer.getAndIncrement() < layers) {
    
                    db.findNodes(LEAF).stream()
                            .peek(node -> node.removeLabel(LEAF))
                            .forEach(node -> {
                                int currentDegree = (int) node.getProperty("degree");
                                while (currentDegree-- > 0) {
                                    Node newNode = db.createNode(NODE, LEAF);
                                    newNode.setProperty("degree", degree.getAndIncrement());
                                    newNode.setProperty("layer", currentLayer.get());
                                    node.createRelationshipTo(newNode, NEXT);
                                }
                            });
                }
    
                tx.success();
            }
        }
    }
}
