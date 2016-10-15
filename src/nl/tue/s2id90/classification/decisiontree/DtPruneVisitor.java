
package nl.tue.s2id90.classification.decisiontree;

import java.util.Set;
import nl.tue.s2id90.classification.data.Features;
import nl.tue.s2id90.classification.data.LabeledDataset2;
import nl.tue.s2id90.classification.labeledtree.LabeledTree;
import nl.tue.s2id90.classification.labeledtree.Visitor;
import nl.tue.s2id90.classification.utils.CardinalityMap;


public class DtPruneVisitor<V extends Features, L> implements Visitor<Object, DecisionTree<V, L>> {
    
    private final LabeledDataset2 testData;
    
    private final DecisionTree root;
    
    public DtPruneVisitor(LabeledDataset2<V, L> testData, DecisionTree root) {
        this.testData = testData;
        this.root = root;
    }

    @Override
    public void visitNode(LabeledTree<Object, DecisionTree<V, L>> node) {
        if (!node.isLeaf()) {            
            Set<L> labels = ((DecisionTree)node).getDataset().getLabels();
            CardinalityMap<L> cardinalityMap = new CardinalityMap<>(labels);
            for (DecisionTree<V, L> subTree : node.getSubTrees()) {
                assert subTree.isLeaf();
                cardinalityMap.incrementCardinality(subTree.getLabel());
            }
            ((DecisionTree)node).setLabel(cardinalityMap.getMaxKey());
            System.out.println(root.errorRate(testData.getClassification()));
        }
    }

    @Override
    public void visitEdge(LabeledTree<Object, DecisionTree<V, L>> parent, DecisionTree<V, L> child, Object label) {
        throw new UnsupportedOperationException("Not supported yet."); 
    }

}
