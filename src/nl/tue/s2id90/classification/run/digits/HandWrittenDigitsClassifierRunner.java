
package nl.tue.s2id90.classification.run.digits;

import java.io.IOException;
import java.util.List;
import nl.tue.s2id90.classification.Classifier;
import nl.tue.s2id90.classification.ConfusionMatrixPanel;
import nl.tue.s2id90.classification.data.LabeledDataset2;
import nl.tue.s2id90.classification.data.digits.DigitsUtil;
import nl.tue.s2id90.classification.data.digits.HandWrittenDigits;
import nl.tue.s2id90.classification.data.digits.datasets.HandWrittenDatasetBits;
import nl.tue.s2id90.classification.data.digits.datasets.HandWrittenDatasetDoubles;
import nl.tue.s2id90.classification.data.digits.features.ImageFeatureType;
import nl.tue.s2id90.classification.decisiontree.DecisionForest;
import nl.tue.s2id90.classification.decisiontree.DecisionTree;
import nl.tue.s2id90.classification.knn.digits.BitsKnnClassifier;
import nl.tue.s2id90.classification.knn.digits.DoublesKnnClassifier;
import nl.tue.s2id90.classification.labeledtree.DotUtil;
import nl.tue.s2id90.classification.utils.ConfusionMatrix;

/**
 *
 * @author iVerb
 * @author Jeroen Noten
 */
public class HandWrittenDigitsClassifierRunner {
    
    private static final int TRAINING_DATA_SIZE_DEFAULT = 15000;
    private static final boolean SHUFFLE_TRAINING_DATA_DEFAULT = true;
    private static final ImageFeatureType FEATURE_TYPE_DEFAULT = ImageFeatureType.DOUBLES;
    
    private static final int SHOW_IMAGE_COUNT_DEFAULT = 0;
    private static final boolean SHOW_CONFUSION_MATRIX_DEFAULT = true;
    private static final boolean SHOW_DECISION_TREE_DEFAULT = false;
    private final boolean APPLY_PRUNING_DEFAULT = true;
    private static final boolean PRINT_ERROR_RATE_DEFAULT = true;
    
    protected int trainingDataSize = TRAINING_DATA_SIZE_DEFAULT;
    protected boolean shuffleTrainingData = SHUFFLE_TRAINING_DATA_DEFAULT;
    protected ImageFeatureType featureType = FEATURE_TYPE_DEFAULT;
    
    protected int showImageCount = SHOW_IMAGE_COUNT_DEFAULT;
    protected boolean showConfusionMatrix = SHOW_CONFUSION_MATRIX_DEFAULT;
    protected boolean showDecisionTree = SHOW_DECISION_TREE_DEFAULT;
    protected boolean applyPruning = APPLY_PRUNING_DEFAULT;
    protected boolean printErrorRate = PRINT_ERROR_RATE_DEFAULT;
    
    
    
    public void runKnnClassifier(int k) throws IOException {
        LabeledDataset2 trainingData = generateTrainingData();
        Classifier classifier = null;
        switch (featureType) {
            case BITS:
                classifier = new BitsKnnClassifier(trainingData.getClassification(), k);
                break;
            case DOUBLES:
                classifier = new DoublesKnnClassifier(trainingData.getClassification(), k);
        }
        runClassifier(classifier);
    }
    
    public void runDecisionTreeClassifier() throws IOException {
        LabeledDataset2 trainingData = generateTrainingData();
        DecisionTree classifier = new DecisionTree(trainingData);
        runClassifier(classifier);
        if (showDecisionTree) {
           DotUtil.showDotInFrame(classifier.toDot(), null);
        }        
        if (applyPruning) {
           LabeledDataset2 testData = generateTestData();
           classifier.prune(testData);
        }       
    }
    
    public void runDtForestClassifier(int k) throws IOException {
        LabeledDataset2 trainingData = generateTrainingData();
        DecisionForest classifier = new DecisionForest(trainingData, k);
        runClassifier(classifier);
        if (applyPruning) {
           LabeledDataset2 testData = generateTestData();
           classifier.prune(testData);        }
        
    }
    
    private void runClassifier(Classifier classifier) throws IOException {
        LabeledDataset2 testData = generateTestData();
        
        ConfusionMatrix confusionMatrix = classifier.getConfusionMatrix(testData.getClassification());
        
        if (showConfusionMatrix) {
            (new ConfusionMatrixPanel(testData, confusionMatrix)).showIt();
        }
        
        if (printErrorRate) {
            System.out.println("Error rate: " + confusionMatrix.getErrorRate());
        }
    }
    
    private LabeledDataset2 generateTrainingData() throws IOException {
        List trainingImages = HandWrittenDigits.getTrainingData(trainingDataSize, shuffleTrainingData);
        return createDataset(trainingImages);
    }
    
    private LabeledDataset2 generateTestData() throws IOException {
        List testImages = HandWrittenDigits.getTestData();

        if (showImageCount > 0) {
            DigitsUtil.showImages("Test data", testImages.subList(0, showImageCount), showImageCount / 10);
        }
        
        return createDataset(testImages);
    }

    private LabeledDataset2 createDataset(List images) throws IOException {
        switch (featureType) {
            case BITS:
                return new HandWrittenDatasetBits(images);
            case DOUBLES:
                return new HandWrittenDatasetDoubles(images);
        }
        return null;
    }

}
