package nl.tue.s2id90.classification.decisiontree;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import nl.tue.s2id90.classification.Classifier;
import nl.tue.s2id90.classification.data.LabeledDataset2;
import nl.tue.s2id90.classification.data.Features;
import nl.tue.s2id90.classification.utils.CardinalityMap;
import nl.tue.s2id90.classification.utils.ConfusionMatrix;

/**
 *
 * @author iVerb
 * 
 * @param <V>
 * @param <T>
 * @see ListUtil#mostOccurringElement(java.util.List)
 */
public class DecisionForest<V extends Features, T> implements Classifier<V, T>{
    
    List<DecisionTree<V,T>> trees = new ArrayList<>();
    
    /**
     * A random forest consisting of k decision trees.
     * 
     * @param dataset
     * @param k number of trees in forest.
     */
    public DecisionForest(LabeledDataset2<V,T> dataset, int k) {
        List<LabeledDataset2<V, T>> datasets = dataset.splitData(k);
        build(datasets);
    }
    
    @Override
    public T classify(V v) { 
        CardinalityMap<T> map = new CardinalityMap<>();
        for (DecisionTree<V, T> dt : trees) {
            T clazz = dt.classify(v);
            if (!map.containsKey(clazz)) 
                map.addKey(clazz);
            map.incrementCardinality(clazz);
        }
        return map.getMaxKey();
    }
    
    /** build k decision trees, one for each data set. **/
    private void build(List<LabeledDataset2<V, T>> datasets) {
        for (LabeledDataset2 dataset : datasets) {
            DecisionTree dt = new DecisionTree(dataset);
            trees.add(dt);
        }
    }    
    
    public void prune(LabeledDataset2<V,T> testData) {
        for (DecisionTree t : trees) {
            t.prune(testData);
        }
    }
    
    @Override
     public double errorRate(Map<V, T> testData) {
        ConfusionMatrix confusionMatrix = getConfusionMatrix(testData);
        return confusionMatrix.getErrorRate();
    }

    @Override
    public ConfusionMatrix getConfusionMatrix(Map<V, T> testData) {
        return new ConfusionMatrix(testData, this);
    }
}
