
package nl.tue.s2id90.classification.knn.digits;

import java.util.Map;
import nl.tue.s2id90.classification.data.digits.features.Doubles;
import nl.tue.s2id90.classification.knn.KnnClassifier;

/**
 * K-nearest neighbor classifier for feature vectors with double type features.
 * 
 * @author iVerb
 * @author Jeroen Noten
 */
public class DoublesKnnClassifier extends KnnClassifier<Doubles, Byte> {

    /**
     * Constructs a classifier with training data and a given K
     * 
     * @param trainingData training data used for classifying test data,
     * maps feature vectors with double type values to classification labels
     * @param k the number of nearest neighbours to consider
     */
    public DoublesKnnClassifier(Map<Doubles, Byte> trainingData, int k) {
        super(trainingData, k);
    }
    
    /**
     * Computes the distance between two feature vectors. We define the distance
     * by the sum of absolute differences between each two features.
     * 
     * @param featureVector1 first feature vector
     * @param featureVector2 second feature vector
     * @return the distance between two feature vectors
     */
    @Override
    public double distance(Doubles featureVector1, Doubles featureVector2) {
        double distance = 0;

        // loop over all features, assuming that both vectors have the same length
        for (int i = 0; i < featureVector1.getNumberOfAttributes(); i++) {
            // add the absolute difference between two features
            distance += Math.abs(featureVector1.get(i) - featureVector2.get(i));
        }
        
        return distance;
    }
    
}
