package ml;

import org.deeplearning4j.graph.models.deepwalk.DeepWalk;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Transaction;

/**
 * @author p.meltzer@braintree.com
 */
public class Deepwalk {

    public static void vectoriseAllNodes(GraphDatabaseService db,
                                         double learningRate,
                                         int rndSeed,
                                         int vectorSize,
                                         int windowSize,
                                         int walkLength, String vectorPropertyName)
            throws InterruptedException {

        DeepwalkGraph deepwalkGraph = new DeepwalkGraph(db);

        DeepWalk<Node, ml.DeepwalkGraph.RelationshipValue> deepWalk = new DeepWalk.Builder<Node, ml.DeepwalkGraph.RelationshipValue>()
                .learningRate(learningRate)
                .seed(rndSeed)
                .vectorSize(vectorSize)
                .windowSize(windowSize)
                .build();

        deepWalk.initialize(deepwalkGraph);
        deepWalk.fit(deepwalkGraph, walkLength);

        try (Transaction tx = db.beginTx()) {

            db.getAllNodes().stream()
                    .forEach(node -> {

                        INDArray vertexVector = deepWalk.getVertexVector(deepwalkGraph.getMappedID(node.getId()));
                        double[] vector = new double[vectorSize];

                        for (int i = 0; i < vector.length; i++)
                            vector[i] = vertexVector.getDouble(i);

                        // add vector for node as a property
                        node.setProperty(vectorPropertyName, vector);
                    });

            tx.success();
        }
    }
}
