package nl.tue.s2id90.classification.data;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import nl.tue.s2id90.classification.decisiontree.Information;

/**
 * Compared to a <code>LabeledDataset</code> this dataset maintains a reversed
 * map, that maps a label on a list of all feature vectors with that label
 * as classification label. Furthermore, this dataset has methods for splitOnAttributeting
 the dataset in parts, and computing the information gain after splitOnAttributeting.
 * @author huub
 * @param <V> feature vector
 * @param <L> classification label
 * @see LabeledDataset
 */
public abstract class LabeledDataset2<V extends Features, L> extends LabeledDataset<V, L> {
    
    protected final Map<L, List<V>> reversedMap; // maps class to list of elements with that class

    public LabeledDataset2() {
        reversedMap = new HashMap<>();
    }
    
    /**
     * @param label 
     * @return the featureVectors classified as label.
     **/
    public List<V> getFeatures(L label) {
        return reversedMap.get(label);
    }
    
    /**
     * adds classified data to this dataset.
     *
     * @param data
     */
    public void putAll(Map<V, L> data) {
        for (Entry<V, L> cd : data.entrySet()) {
            put(cd.getValue(),cd.getKey());
        }
    }
    
    /**
     * adds classified data to this dataset.
     *
     * @param t     classification
     * @param data  collection of feature vectors
     */
    public void putAll(L t, Collection<V> data) {
        for (V v : data) {
            put(t, v);
        }
    }

    /**
     * adds a single feature vector V and it's classification T to this dataset.
     *
     * @param t classification class
     * @param v feature vector
     */
    public void put(L t, V v) {
        checkDimensionality(v);
        List<V> list = reversedMap.get(t);
        if (list == null) {
            list = new ArrayList<>();
            reversedMap.put(t, list);
        }
        list.add(v);
        classification.put(v, t);
    }
    

    /** splitOnAttributes dataset in subsets with a constant value for the i-th feature.
     * @param i
     * @return result of splitOnAttributeting, the keys in this map are the values of the i-th
        attribute.
     **/    
    public Map<AttributeSplit,LabeledDataset2<V, L>> splitOnAttribute(int i) {
        Map<AttributeSplit, LabeledDataset2<V, L>> map = new HashMap<>();
        for (AttributeSplit split : getAttributeSplits(i)) {
            LabeledDataset2 splitDataset = createDataset();
            for (V features : featureVectors()) {
            L label = getLabel(features);            
                if (conformsToSplit(split, features.get(i))) {
                    splitDataset.put(label, features);
                }
            }
            map.put(split, splitDataset);
        }
        return map;
    }
    
    /**
     * 
     * @param k
     * @return 
     */
    public List<LabeledDataset2<V, L>> splitData(int k) {
        List<LabeledDataset2<V, L>> datasets = new ArrayList<>();
        Iterator<Map.Entry<V, L>> iterator = classification.entrySet().iterator();
        int splitSize = size() / k;
        int rest = size() % splitSize;
        for (int i = 0; i < k; i++) {
            LabeledDataset2<V, L> dataset = createDataset();
            for (int d = 0; d < splitSize; d++) {
                Map.Entry<V, L> entry = iterator.next();
                dataset.put(entry.getValue(), entry.getKey());
            }
            datasets.add(dataset);
        }
        for (int d = 0; d < rest; d++) {
            Map.Entry<V, L> entry = iterator.next();
            datasets.get(k - 1).put(entry.getValue(), entry.getKey());
         }
        return datasets;
    }   
        
    /** returns the probabilities of the classification labels for this dataset.
     *  @return probability of the labels.
     **/
    public double getInitialGain() {
        double[] distribution = new double[getLabels().size()];
        int i = 0;
        for (L label1 : reversedMap.keySet()) {
            double numOccurrences = 0;
            for (L label2 : classification.values()) {
                if (label2 == label1)
                    numOccurrences++;
            }
            distribution[i] = numOccurrences / size();
            i++;
        }
        return Information.entropy(distribution);
    }
    
    /**
     * 
     * @param i
     * @param split
     * @return 
     */
    private double getSplitGain(int i, AttributeSplit split) {
        double[] distribution = new double[getLabels().size()];
        double splitSize = 0;
        int c = 0;
        for (L label : reversedMap.keySet()) {
            double numOccurrences = 0;
            for (V f : featureVectors()) {
                if (conformsToSplit(split, f.get(i))) {
                    if (c == 0) 
                        splitSize++;
                    if (getLabel(f) == label)
                        numOccurrences++;
                }
            }
            if (splitSize == 0) return 0;
            else distribution[c] = numOccurrences / splitSize;
            c++;
        }
        double scalar = splitSize / size();
        return scalar * Information.entropy(distribution);
    }
    
    /**
     * @param i index of attribute used for continuous splitOnAttributeting
     * @return gain by splitOnAttributeting on attribute i.
     */
    public double gain(int i) {
        double initialGain = getInitialGain();
        double splitGain = 0;
        for (AttributeSplit split : getAttributeSplits(i)) {
            splitGain += getSplitGain(i, split);
        }
        //System.out.println(initialGain - splitGain);
        return initialGain - splitGain;        
    }
    
    /**
     * 
     * @param i
     * @return 
     */
    public Set<AttributeSplit> getAttributeSplits(int i) {
        Set<AttributeSplit> splits = new HashSet<>();
        if (isContinuousFeature(i)) {
            Number splitValue = (Number)featureVectors().get(0).get(i);
            splits.add(new AttributeSplit(i, splitValue, true));
            splits.add(new AttributeSplit(i, splitValue, false));
        } else {
            Set<Object> discreteValues = new HashSet<>();
            for (V f : featureVectors()) {
                Object value = f.get(i);
                if (!discreteValues.contains(value)) {
                    discreteValues.add(value);
                    splits.add(new AttributeSplit(i, value));
                }
            }
        }
        return splits;
    }
    
    /**
     *
     * @param value
     * @param split
     * @return
     */
    public boolean conformsToSplit(AttributeSplit split, Object value) {
        if (isContinuousFeature(split.index)) {
            double doubleValue = ((Number)value).doubleValue();
            double doubleSplitValue = ((Number)split.value).doubleValue();
            return split.upperBound && doubleValue <= doubleSplitValue
                    || !split.upperBound && doubleValue > doubleSplitValue;
        }
        else return split.value == value;
    }
    
     /** @return the most frequently occurring class in this dataset;
      * null, if the dataset is empty.
     */
    public L getMostFrequentClass() {
        L maxLabel = null;
        int maxFrequency = 0;
        for (L t : getLabels()) {
            int labelFrequency = getFeatures(t).size();
            if (labelFrequency > maxFrequency) {
                maxLabel = t;
                maxFrequency = labelFrequency;
            }
        }
        return maxLabel;       
    }
    
    /** @return the frequencies of the subclasses . **/
    public List<Integer> getFrequencies() {
        List<Integer> result = new ArrayList<>();
        for (L t : getLabels()) {
            result.add(getFeatures(t).size());
        }
        return result;
    }

    /**
     * 
     * @return 
     */
    public abstract LabeledDataset2<V, L> createDataset();

}