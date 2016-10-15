
package nl.tue.s2id90.classification.utils;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import nl.tue.s2id90.classification.Classifier;
import nl.tue.s2id90.classification.data.Features;

/**
 *
 * @author iVerb
 * @param <F>
 * @param <L>
 * @since
 */
public class ConfusionMatrix<F extends Features, L> extends HashMap<L, CardinalityMap<L>> {
    
    public ConfusionMatrix(Map<F, L> testData, Classifier<F, L> classifier) {
        for (L label : testData.values()) {
            CardinalityMap<L> classification = new CardinalityMap<>(testData.values());
            put(label, classification);
        }
        for (F v : testData.keySet()) {
            L label = testData.get(v);
            L result = classifier.classify(v);
            get(label).incrementCardinality(result);
        }
    }
    
    public double getErrorRate() {
        int errorCount = 0;
        int size = 0;
        for (Entry<L, CardinalityMap<L>> row : entrySet()) {
            L label = row.getKey();
            CardinalityMap<L> map = row.getValue();
            for (Entry<L, Integer> cardinality : map.entrySet()) {
                if (cardinality.getKey() != label) {
                    errorCount += cardinality.getValue();
                }
                size += cardinality.getValue();
            }
        }
        return (double) errorCount / size;
    }
    
}
