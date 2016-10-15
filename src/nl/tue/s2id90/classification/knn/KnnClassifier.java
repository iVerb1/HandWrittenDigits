package nl.tue.s2id90.classification.knn;

import java.util.Collection;
import java.util.Comparator;
import java.util.Map;
import java.util.TreeMap;
import nl.tue.s2id90.classification.Classifier;
import nl.tue.s2id90.classification.data.Features;
import nl.tue.s2id90.classification.utils.CardinalityMap;
import nl.tue.s2id90.classification.utils.ConfusionMatrix;

/**
 * K-nearest neighbor classifier.
 * <pre>
 *{@code 
 *            // read handwritten digits training data
            List<LabeledImage> trainingData = HandWrittenDigits.getTrainingData(15000, true);
            Map<ImageFeatures<Double>,Byte> trainingDataset = new HashMap<>();
            for(LabeledImage image : trainingData) {
                  trainingDataset.put(new Doubles(image),image.getLabel());
            }
            KnnClassifier knn = new KnnClassifier<ImageFeatures<Double>,Byte>(trainingData);
            ...
}
 * </pre>

 * @param <F> feature vector
 * @param <L> classification label
 * 
 * @author iVerb
 * @author Jeroen Noten
 */
public abstract class KnnClassifier<F extends Features, L> implements Classifier<F, L>  {
    
    // training data used for classifying test data
    protected Map<F, L> trainingData;
    
    // the number of nearest neighbours to consider
    protected int k;
    
    /**
     * Constructs a classifier with training data and a given K
     * 
     * @param trainingData training data used for classifying test data,
     * maps feature vectors to classification labels
     * @param k the number of nearest neighbours to consider
     */
    public KnnClassifier(Map<F, L> trainingData, int k) {
        this.trainingData = trainingData;
        this.k = k;
    }

    /**
     * computes the distance between two feature vectors. This distance is
     * used by the nearest neighbor algorithm.
     * <p>
     * A typical implementation extends this class to implement this method.
     * For instance, using inner classes :
     * <pre>
     *{@code 
     *             // create a KnnClassifier object for features <code>MyFeatures</code>
             // and Label type MyLabel.
             KnnClassifier classifier = new KnnClassifier<MyFeatures,MyLabel>(trainingDataset) {
                 // give an implementation of the distance function
                 public double distance(MyFeatures f0, MyFeatures f1) {
                     // compute the distances between f0 and f1
                     // using f0.get(i), and f1.get(i)
                 }
             };
     
}
     *</pre>
     * Or a separate class: <pre>
     *{@code 
     *             public class MyKnnClassifier extends KnnClassifier<MyFeatures,MyLabel> {
                  public MyKnnClassifier(Map<MyFeatures,MyLabel> trainingData) {
                       super(trainingData);
                  }
                 // give an implementation of the distance function
                 public double distance(MyFeatures f0, MyFeatures f1) {
                     // compute the distances between f0 and f1
                     // using f0.get(i), and f1.get(i)
                 }
             };     
}
     * </pre>
     * @param featuresVector1 first feature vector
     * @param featuresVector2 second feature vector
     * @return distance between the two features
     **/
    protected abstract double distance(F featuresVector1, F featuresVector2);

    /**
     * Classifies a features vector based on the training data.
     * 
     * @param featuresVector features vector to classify
     * @return classification label
     */
    @Override
    public L classify(F featuresVector) {
        // find the k nearest neighbors
        final Collection<L> nearestNeighbors = getNearestNeighbors(featuresVector);
        // return the label that is most frequent
        return getMostFrequentLabel(nearestNeighbors);
    }

    /**
     * Determines the error rate of the test data classification.
     * 
     * @param testData test data to classify,
     * maps feature vectors to the real classification labels
     * @return error rate
     */
    @Override
    public double errorRate(Map<F, L> testData) {
        // We build a confusion matrix in order to use the error rate function
        // that is already in the ConfusionMatrix class
        ConfusionMatrix<F, L> confusionMatrix = getConfusionMatrix(testData);
        return confusionMatrix.getErrorRate();
    }
    
    /**
     * Calculates the confusion matrix of the test data classification.
     * 
     * @param testData test data to classify,
     * maps feature vectors to the real classification labels
     * @return confusion matrix
     */
    @Override
    public ConfusionMatrix<F, L> getConfusionMatrix(Map<F, L> testData) {
        return new ConfusionMatrix<>(testData, this);
    }
    
    /**
     * Find the (at most) k nearest neighbors in the set of training data.
     * 
     * @param featuresVector feature vector
     * @return map of at most k nearest feature vector to classification labels
     */
    private Collection<L> getNearestNeighbors(final F featuresVector) {
        // we use a tree map here, in order to keep the map sorted on distance;
        // we feed the features vector to the distance comparator in order to
        // compare all distances to the feature vector that we are investigating
        TreeMap<F, L> nearestNeighbors = new TreeMap<>(new DistanceComparator(featuresVector));
        
        // loop over the full set of training data
        for (Map.Entry<F, L> neighbor : trainingData.entrySet()) {
            F neighborFeatures = neighbor.getKey();
            L neighborLabel = neighbor.getValue();
            
            // just put the neighbor in the set
            nearestNeighbors.put(neighborFeatures, neighborLabel);
            
            // make sure that we have not more than k elements in the map,
            // by removing the element with the greatest distance;
            // we could have done this after the for-loop, but then inserting
            // new feature vectors in the map cost more time, because it is a
            // sorted map
            if (nearestNeighbors.size() > k) {
                nearestNeighbors.pollLastEntry();
            }
        }
        // return just a collection of labels, we do not need the feature vectors anymore
        return nearestNeighbors.values();
    }
    
    /**
     * Find the label that is most frequent in the given map of feature fectors.
     * 
     * @param labels a collection of labels
     * @return most frequent label
     */
    private L getMostFrequentLabel(Collection<L> labels) {
        // we build a cardinality map in order to make counting easy.
        CardinalityMap<L> labelCardinalityMap = new CardinalityMap<>();
        // the most frequent label we have found so far
        L mostFrequentLabel = null;
        // the frequency so far of the most frequent label we have found so far
        int cardinalityMax = 0;

        for (L label : labels) {
            int cardinality = labelCardinalityMap.incrementCardinality(label);
            
            if (cardinality > cardinalityMax) {
                // update the most frequent label and coresponding frequency
                // because the current label is found more frequent than the
                // most frequent label we have found so far
                cardinalityMax = cardinality;
                mostFrequentLabel = label;
            }
        }
        
        return mostFrequentLabel;
    }
    
    /**
     * Comparator that compares the distances to a certain feature vector.
     */
    private class DistanceComparator implements Comparator<F> {
        
        // target feature vector
        private final F featuresVector;
        
        /**
         * Construct a new distance comparator with a target feature vector
         * 
         * @param featuresVector target feature vector
         */
        public DistanceComparator(F featuresVector) {
            this.featuresVector = featuresVector;
        }
        
        /**
         * Compare the distances of feature vectors.
         * 
         * @param featuresVector1 first feature vector
         * @param featuresVector2 second feature vector
         * @return comparison value
         */
        @Override
        public int compare(F featuresVector1, F featuresVector2) {
            // calculate for both feature vectors the distance to the target feature vector
            double distance1 = distance(featuresVector, featuresVector1);
            double distance2 = distance(featuresVector, featuresVector2);
            // just regularly compare the distances
            return Double.compare(distance1, distance2);
        }

    }
}