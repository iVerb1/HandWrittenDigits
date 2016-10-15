package nl.tue.s2id90.classification.run.digits;

import java.io.IOException;
import nl.tue.s2id90.classification.data.Features;
import nl.tue.s2id90.classification.data.LabeledDataset2;
import nl.tue.s2id90.classification.data.golf.GolfData;
import nl.tue.s2id90.classification.data.golf.GolfData.PLAY;
import nl.tue.s2id90.classification.data.golf.GolfDataset;
import nl.tue.s2id90.classification.decisiontree.DecisionTree;
import nl.tue.s2id90.classification.labeledtree.DotUtil;

/**
 *
 * @author Jeroen Noten
 * @author iVerb
 */
public class HandWrittenDigitsDtClassifierRunner extends HandWrittenDigitsClassifierRunner {
    
    private static final int K = 15;
    
    private HandWrittenDigitsDtClassifierRunner() {
        trainingDataSize = 15000;
        showDecisionTree = false;
    }
    
    public static void main(String[] args) throws IOException {
        (new HandWrittenDigitsDtClassifierRunner()).runDecisionTreeClassifier();
        
        //LabeledDataset2<Features, PLAY> dataset = new GolfDataset(new GolfData());
        //DecisionTree dt = new DecisionTree(dataset);
        //DotUtil.showDotInFrame((dt).toDot(), null);
        //dt.prune(dataset);

    }
    
}
