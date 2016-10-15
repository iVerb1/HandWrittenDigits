package nl.tue.s2id90.classification.run.digits;

import java.io.IOException;

/**
 *
 * @author Jeroen Noten
 * @author iVerb
 */
public class HandWrittenDigitsKnnClassifierRunner extends HandWrittenDigitsClassifierRunner {
    
    private static final int K = 5;
    
    private HandWrittenDigitsKnnClassifierRunner() {
        trainingDataSize = 100;
    }
    
    public static void main(String[] args) throws IOException {
        (new HandWrittenDigitsKnnClassifierRunner()).runKnnClassifier(K);
    }
    
}
