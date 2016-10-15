
package nl.tue.s2id90.classification.utils;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map.Entry;

/**
 *
 * @author jeroennoten
 */
public class CardinalityMap<K> extends HashMap<K, Integer> {
    
    public CardinalityMap(Collection<K> keys) {
        for (K key : keys) {
            addKey(key);
        }
    }

    public CardinalityMap() {
    }
    
    public void addKey(K key) {
        put(key, 0);
    }
    
    /**
     * 
     * @param key
     * @return 
     */
    public int incrementCardinality(K key) {
        Integer cardinality = get(key);
        if (cardinality == null) {
            cardinality = 0;
        }
        cardinality++;
        put(key, cardinality);
        return cardinality;
    }
    
    /**
     * 
     * @param key
     * @return 
     */
    public int getCardinality(K key) {
        Integer cardinality = get(key);
        if (cardinality == null) {
            cardinality = 0;
        }
        return cardinality;
    }
    
    /**
     * 
     * @return 
     */
    public K getMaxKey() {
        int maxCardinality = 0;
        K maxKey = null;
        for (Entry<K, Integer> e : entrySet()) {
            if (e.getValue() > maxCardinality) {
                maxKey = e.getKey();
                maxCardinality = e.getValue();
            }
        }
        return maxKey;        
    }
    
}
