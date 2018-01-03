package ml;

import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Label;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.RelationshipType;
import org.neo4j.procedure.Context;
import org.neo4j.procedure.Mode;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.Procedure;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.net.URL;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Stream;

/**
 * @author mh
 * @since 26.07.17
 * see: https://www.tensorflow.org/extend/tool_developers/
     * see: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto
 * https://developers.google.com/protocol-buffers/docs/javatutorial
 * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/saved_model/testdata/half_plus_two/00000123/saved_model.pb
 *
 */
public class LoadTensorFlow {

    @Context
    public GraphDatabaseService db;

    enum Types implements Label {
        Neuron
    }

    enum RelTypes implements RelationshipType {
        INPUT
    }

    public static class LoadResult {
        public String modelName;
        public String type;
        public long nodes;
        public long relationships;

        public LoadResult(String modelName, String type, long nodes, long relationships) {
            this.modelName = modelName;
            this.type = type;
            this.nodes = nodes;
            this.relationships = relationships;
        }
    }
    @Procedure(value = "load.tensorflow", mode = Mode.WRITE)
    public Stream<LoadResult> loadTensorFlow(@Name("file") String url) throws IOException {
        GraphDef graphDef = GraphDef.parseFrom(new BufferedInputStream(new URL(url).openStream()));
        Map<String, Node> nodes = new HashMap<>();
        // tod model node, layer nodes
        for (NodeDef nodeDef : graphDef.getNodeList()) {
            Node node = db.createNode(Types.Neuron);
            node.setProperty("name", nodeDef.getName());
            if (nodeDef.getDevice() != null) node.setProperty("device", nodeDef.getDevice());
            node.setProperty("op", nodeDef.getOp());
            nodeDef.getAttrMap().forEach((k, v) -> {
                Object value = getValue(v);
                if (value != null) {
                    node.setProperty(k, value);
                }
            });
            nodes.put(nodeDef.getName(), node);
        }
        long rels = 0;
        for (NodeDef nodeDef : graphDef.getNodeList()) {
            Node target = nodes.get(nodeDef.getName());
            nodeDef.getInputList().forEach(name -> nodes.get(name).createRelationshipTo(target, RelTypes.INPUT));
            // todo weights
            rels += nodeDef.getInputCount();
        }
        return Stream.of(new LoadResult(url,"tensorflow",nodes.size(), rels));
    }

    private Object getValue(AttrValue v) {
        switch (v.getValueCase()) {
            case S:
                return v.getS().toStringUtf8();
            case I:
                return v.getI();
            case F:
                return v.getF();
            case B:
                return v.getB();
            case TYPE:
                return v.getType().name(); // todo
            case SHAPE:
                return v.getShape().toString(); // tdo
            case TENSOR:
                return v.getTensor().toString(); // todo handle with prefxied properties
            case LIST:
                return v.getList().toString(); // todo getType/Count(idx) and then handle each type with prefixed property
            case FUNC:
                return v.getFunc().getAttrMap().toString(); // todo handle recursively
            case PLACEHOLDER:
                break;
            case VALUE_NOT_SET:
                return null;
            default:
                return null;
        }
        return null;
    }

}
