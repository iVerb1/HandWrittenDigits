package nl.tue.s2id90.classification.decisiontree;

import nl.tue.s2id90.classification.labeledtree.LabeledTree;
import java.util.Map;
import nl.tue.s2id90.classification.Classifier;
import nl.tue.s2id90.classification.data.LabeledDataset2;
import nl.tue.s2id90.classification.data.Features;
import nl.tue.s2id90.classification.data.AttributeSplit;
import nl.tue.s2id90.classification.labeledtree.Visitor;
import nl.tue.s2id90.classification.utils.ConfusionMatrix;

/**
 *
 * 
 * @param <F> feature type
 * @param <L> label type
 * 
 * @author iVerb
 */
public class DecisionTree<F extends Features,L> 
    extends LabeledTree<Object,DecisionTree<F,L>> implements Classifier<F,L> {    
    
    protected L label;
    protected int splitAttribute;
    protected double splitGain;
    
    
    /** All the feature vectors in this tree. **/
    final protected LabeledDataset2<F,L> dataset;
    
    /** constructs a decision tree with the given dataset as leaves
     * @param dataset
     */
    public DecisionTree(LabeledDataset2<F,L> dataset) {
        label = null;
        this.dataset = dataset;
        build(dataset);
    }
    
    /** creates a leave containing the given dataset and classified as clazz.
     * @param dataset
     * @param clazz
     **/
    public DecisionTree(LabeledDataset2 dataset, L clazz) {
        assert clazz!=null;
        this.dataset = dataset;
        this.label = clazz;
    }
    
    public LabeledDataset2<F,L> getDataset() {
        return dataset;
    }
    
    public L getLabel() {
        return label;
    }
    
    public void setLabel(L label) {
        this.label = label;
    }
    
    public int getSplitAttribute() {
        return splitAttribute;
    }

    
    /** sets index of attribute on which this tree will splitOnAttribute on the root level.
     * If splitOnAttributeValue!=null, it indicates a continuous splitOnAttribute of the attribute with this index,
 and its features have been splitOnAttribute in : smaller or equal to splitOnAttributeValue and larger than splitOnAttributeValue.
     * 
     * @param index 
     * @param splitValue 
     * @param gain 
     */
    public void setSplit(int index, double gain) {
        splitAttribute = index;
        this.splitGain = gain;
    }
    
    
    public int getSize() {
        return dataset!=null?dataset.size():0;
    }
    
    /** to facilitate the pruning mechanism, we will consider nodes with a 
     * non-null label, as leaf nodes.
     * @return 
     */
    @Override
    public boolean isLeaf() {
        if (label!=null) return true;
        else return super.isLeaf();
    }
    
    @Override
    public L classify(F v) {
        if (isLeaf())
            return getLabel();
        Object value = v.get(splitAttribute);
        AttributeSplit split = null;
        for (AttributeSplit s : dataset.getAttributeSplits(splitAttribute)) {
            if (dataset.conformsToSplit(s, value))
                split = s;
        }
        for (DecisionTree<F, L> t : getSubTrees()) {
            F subTreeFeatures = t.getDataset().featureVectors().get(0);            
            Object valueInSubTree = subTreeFeatures.get(splitAttribute);
            if (t.getDataset().conformsToSplit(split, valueInSubTree))
                return t.classify(v);
        }
        throw new IllegalArgumentException();
    }
     
    public int prune(final LabeledDataset2<F,L> testData) {
        Visitor pruneVisitor = new DtPruneVisitor(testData, this);
        breadthFirstPostOrderVisit(pruneVisitor);
        return 0;
    }

    @Override
    public ConfusionMatrix getConfusionMatrix(Map<F, L> testData) {
        return new ConfusionMatrix(testData, this);
    }
    
    @Override
    public double errorRate(Map<F, L> testData) {
        ConfusionMatrix confusionMatrix = getConfusionMatrix(testData);
        return confusionMatrix.getErrorRate();
    }
        
    /**
     *
     * @param dataset
     */
    protected void build(LabeledDataset2<F,L> dataset) {
        getMaxGainAttribute(dataset);
        System.out.println(splitGain);
        if (splitGain == 0) 
                setLabel(dataset.getMostFrequentClass());
        else {
            Map<AttributeSplit, LabeledDataset2<F, L>> children = dataset.splitOnAttribute(splitAttribute);
            for (AttributeSplit split : children.keySet()) {
                LabeledDataset2 childData = children.get(split);
                configureSubTree(split, childData);
            } 
        }
    }
    
    private void getMaxGainAttribute(LabeledDataset2<F,L> dataset) {
        double maxGain = 0;
        int maxAttribute = 0;
        for (int i = 0; i < dataset.getNumberOfDimensions(); i ++) {
            double gain = dataset.gain(i);          
            if (gain > maxGain) {
                maxGain = gain;
                maxAttribute = i;
            }
        }
        setSplit(maxAttribute, maxGain);
    }
    
    private void configureSubTree(AttributeSplit edgeLabel, LabeledDataset2<F,L> dataset) {
        DecisionTree tree;
        if (dataset.getLabels().size() == 1) 
            tree = new DecisionTree(dataset, dataset.getMostFrequentClass());
        else 
            tree = new DecisionTree(dataset);
        addSubTree(edgeLabel, tree);
    }
        
    @Override
    public String toDot() {
        final StringBuilder b = new StringBuilder("digraph DT {\n");
        DecisionTreeDotVisitor<F,L> visitor = new DecisionTreeDotVisitor<>(b);                
        depthFirstPreOrderVisit(visitor);
        return b.append("}").toString();
    }

}