package nl.tue.s2id90.classification.knn.digits;

import java.util.Map;
import nl.tue.s2id90.classification.data.digits.features.Bits;
import nl.tue.s2id90.classification.knn.KnnClassifier;

/**
 * K-nearest neighbor classifier for feature vectors with binary features.
 * 
 * @author iVerb
 * @author Jeroen Noten
 */
public class BitsKnnClassifier extends KnnClassifier<Bits, Byte> {
    
    /**
     * Constructs a classifier with training data and a given K
     * 
     * @param trainingData training data used for classifying test data,
     * maps feature vectors with binary values to classification labels
     * @param k the number of nearest neighbours to consider
     */
    public BitsKnnClassifier(Map<Bits, Byte> trainingData, int k) {
        super(trainingData, k);
    }
    
    /**
     * Computes the distance between two feature vectors. We define the distance
     * by the number of different feature vectors.
     * 
     * @param featureVector1 first feature vector
     * @param featureVector2 second feature vector
     * @return the distance between two feature vectors
     */
    @Override
    protected double distance(Bits featureVector1, Bits featureVector2) {
        // we use a static method to improve testability
        return calculateDistance(featureVector1, featureVector2);
    }
    
    /**
     * Computes the distance between two feature vectors. We define the distance
     * by the number of different feature vectors.
     * 
     * @param featureVector1 first feature vector
     * @param featureVector2 second feature vector
     * @return the distance between two feature vectors
     */
    public static double calculateDistance(Bits featureVector1, Bits featureVector2) {
        int distance = 0;
        
        // loop over all features, assuming that both vectors have the same length
        for (int i = 0; i < featureVector1.getNumberOfAttributes(); i++){
            byte bit1 = featureVector1.get(i);
            byte bit2 = featureVector2.get(i);
            
            if (bit1 != bit2) {
                // increment the distance if the bits are not equal to each other
                distance++;
            }
        }
        
        return distance;
    }
    
}
