package ml;

import org.deeplearning4j.graph.api.BaseGraph;
import org.deeplearning4j.graph.api.Edge;
import org.deeplearning4j.graph.api.IGraph;
import org.deeplearning4j.graph.api.Vertex;
import org.deeplearning4j.graph.exception.NoEdgesException;
import org.deeplearning4j.graph.models.deepwalk.DeepWalk;
import org.neo4j.graphdb.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * An implementation of {@link IGraph} that connects directly to
 * an existing neo4j graph database, to allow use of {@link DeepWalk}.
 *
 * All node IDs are mapped to integers in the range 0 to nVertices - 1. When requesting the individual
 * vectors using {@code deepwalk.getVertexVector} method directly, the mapped ID should be used, e.g.
 *
 * <pre>
 * {@code
 *
 * DeepWalk<Node, ml.DeepwalkGraph.RelationshipValue> deepWalk =
 *          new DeepWalk.Builder<Node, ml.DeepwalkGraph.RelationshipValue>()
 *               .learningRate(learningRate)
 *               .seed(rndSeed)
 *               .vectorSize(vectorSize)
 *               .windowSize(windowSize)
 *               .build();
 *
 * deepWalk.initialize(deepwalkGraph);
 * deepWalk.fit(deepwalkGraph, walkLength);
 *
 * ...
 *
 * // Get mapped id here
 * INDArray vertexVector = deepWalk.getVertexVector(deepwalkGraph.getMappedID(node.getId()));
 * }
 * </pre>
 *
 * @author p.meltzer@braintree.com
 */
public class DeepwalkGraph extends BaseGraph<Node, DeepwalkGraph.RelationshipValue> {

    private Logger log = LoggerFactory.getLogger(DeepwalkGraph.class);
    private GraphDatabaseService db;
    private long[] graphIDtoNodeIDs;
    private Map<Long, Integer> nodeIDtoGraphIDMap;

    public DeepwalkGraph(GraphDatabaseService db) {
        this.db = db;

        int size = numVertices();
        nodeIDtoGraphIDMap = new ConcurrentHashMap<>(size);

        try (Transaction tx = db.beginTx()) {

            log.info("Creating array of ids...");
            graphIDtoNodeIDs = db.getAllNodes().stream()
                    .mapToLong(Node::getId)
                    .toArray();

            tx.success();

            // split array of IDs for concurrent processing

            int nCPUs = Runtime.getRuntime().availableProcessors();
            log.info("Creating maps with {} threads...", nCPUs);

            Thread[] threads = new Thread[nCPUs];
            int chunkSize = (int) Math.ceil((double) graphIDtoNodeIDs.length / (double) nCPUs);
            int remaining = graphIDtoNodeIDs.length;
            int[] start = new int[threads.length];
            int[] stop = new int[threads.length];
            int current = 0;

            // calculate start and end indexes for each thread
            for (int i = 0; i < threads.length; i++) {

                    chunkSize = Math.min(chunkSize, remaining);

                    start[i] = current;
                    stop[i] = current + chunkSize;

                    current = stop[i];
                    remaining -= chunkSize;
            }

            // create threads
            for (int t = 0; t < threads.length; t++) {

                final int tID = t;
                threads[t] = new Thread(() -> {

                    for (int i = start[tID]; i < stop[tID]; i++) {
                        nodeIDtoGraphIDMap.put(graphIDtoNodeIDs[i], i);
                    }
                });

                threads[t].start();
            }

            for (Thread t : threads) t.join();

            log.info("Completed.");

        } catch (InterruptedException e) {
            e.printStackTrace();
        }

    }

    @Override
    public int numVertices() {

        try (Transaction tx = db.beginTx()) {
            int count = (int) db.getAllNodes().stream().count();
            tx.success();
            return count;
        }
    }

    @Override
    public Vertex<Node> getVertex(int idx) {

        try (Transaction tx = db.beginTx()) {
            Node node = db.getNodeById(graphIDtoNodeIDs[idx]);
            Vertex<Node> nodeVertex = new Vertex<>(idx, node);
            tx.success();
            return nodeVertex;
        }
    }

    @Override
    public List<Vertex<Node>> getVertices(int[] indexes) {

        try (Transaction tx = db.beginTx()) {
            List<Vertex<Node>> list = Arrays.stream(indexes)
                    .mapToLong(i -> graphIDtoNodeIDs[i])
                    .mapToObj(db::getNodeById)
                    .map(node -> new Vertex<>(nodeIDtoGraphIDMap.get(node.getId()), node))
                    .collect(Collectors.toList());
            tx.success();
            return list;
        }
    }

    @Override
    public List<Vertex<Node>> getVertices(int from, int to) {

        try (Transaction tx = db.beginTx()) {
            List<Vertex<Node>> list = Stream.iterate(from, integer -> integer + 1)
                    .limit(to - from + 1)
                    .mapToLong(i -> graphIDtoNodeIDs[i])
                    .mapToObj(db::getNodeById)
                    .map(node -> new Vertex<>(nodeIDtoGraphIDMap.get(node.getId()), node))
                    .collect(Collectors.toList());
            tx.success();
            return list;
        }
    }

    @Override
    public void addEdge(Edge edge) {

        try (Transaction tx = db.beginTx()) {
            Node from = db.getNodeById(graphIDtoNodeIDs[edge.getFrom()]);
            Node to = db.getNodeById(graphIDtoNodeIDs[edge.getTo()]);

            RelationshipValue value = (RelationshipValue) edge.getValue();

            Relationship rel = from.createRelationshipTo(to, value.getType());
            value.getProperties().forEach(rel::setProperty);
            tx.success();
        }
    }

    @Override
    public List<Edge<RelationshipValue>> getEdgesOut(int vertex) {

        long nodeId = graphIDtoNodeIDs[vertex];

        try (Transaction tx = db.beginTx()) {
            List<Edge<RelationshipValue>> list = ((ResourceIterator<Relationship>)
                    db.getNodeById(nodeId).getRelationships(Direction.OUTGOING).iterator()).stream()
                    .map(rel ->
                            new Edge<>(vertex,
                                    nodeIDtoGraphIDMap.get(rel.getEndNode().getId()),
                                    new RelationshipValue(
                                            rel.getType(),
                                            rel.getAllProperties()),
                                    true))
                    .collect(Collectors.toList());
            tx.success();
            return list;
        }
    }

    @Override
    public int getVertexDegree(int vertex) {

        try (Transaction tx = db.beginTx()) {
            int id = db.getNodeById(graphIDtoNodeIDs[vertex]).getDegree();
            tx.success();
            return id;
        }
    }

    @Override
    public Vertex<Node> getRandomConnectedVertex(int vertex, Random rng) throws NoEdgesException {

        try (Transaction tx = db.beginTx()) {
            Node node = db.getNodeById(graphIDtoNodeIDs[vertex]);

            if (node.getDegree() == 0) throw new NoEdgesException("Vertex " + vertex + " has no edges.");

            AtomicInteger count = new AtomicInteger(2);
            Node randomNode = ((ResourceIterator<Relationship>)
                    node.getRelationships().iterator()).stream()
                    .map(rel -> rel.getOtherNode(node))
                    .reduce(node, (acc, next) ->
                            rng.nextDouble() > 1.0 / count.getAndIncrement() ? acc : next);

            Vertex<Node> nodeVertex = new Vertex<>(nodeIDtoGraphIDMap.get(randomNode.getId()), randomNode);
            tx.success();
            return nodeVertex;
        }
    }

    @Override
    public List<Vertex<Node>> getConnectedVertices(int vertex) {

        try (Transaction tx = db.beginTx()) {
            Node node = db.getNodeById(graphIDtoNodeIDs[vertex]);

            List<Vertex<Node>> list = ((ResourceIterator<Relationship>)
                    node.getRelationships().iterator()).stream()
                    .map(rel -> rel.getOtherNode(node))
                    .map(other -> new Vertex<>(nodeIDtoGraphIDMap.get(other.getId()), other))
                    .collect(Collectors.toList());
            tx.success();
            return list;
        }
    }

    @Override
    public int[] getConnectedVertexIndices(int vertex) {

        try (Transaction tx = db.beginTx()) {
            Node node = db.getNodeById(graphIDtoNodeIDs[vertex]);

            int[] integers = ((ResourceIterator<Relationship>)
                    node.getRelationships().iterator()).stream()
                    .map(rel -> rel.getOtherNode(node))
                    .mapToInt(other -> nodeIDtoGraphIDMap.get(other.getId()))
                    .toArray();
            tx.success();
            return integers;
        }
    }

    public int getMappedID(Long neo4jNodeID) {
        return nodeIDtoGraphIDMap.get(neo4jNodeID);
    }

    public static class RelationshipValue {

        private RelationshipType type;
        private Map<String, Object> properties;

        RelationshipValue(RelationshipType type, Map<String, Object> properties) {
            this.type = type;
            this.properties = properties;
        }

        RelationshipType getType() {
            return type;
        }

        Map<String, Object> getProperties() {
            return properties;
        }

        public Object getProperty(String key) {
            return properties.get(key);
        }
    }
}
