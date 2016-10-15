
package nl.tue.s2id90.classification.data.golf;

import nl.tue.s2id90.classification.data.Features;
import nl.tue.s2id90.classification.data.LabeledDataset2;

/**
 *
 * @author iVerb
 */
public class GolfDataset extends LabeledDataset2<Features,GolfData.PLAY> {

    public GolfDataset() {
    }
    
    public GolfDataset(GolfData golfData) {
        putAll(golfData.getClassification());
    }

    @Override
    public LabeledDataset2<Features, GolfData.PLAY> createDataset() {
        return new GolfDataset();
    }
    
}
