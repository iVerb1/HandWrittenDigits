
package nl.tue.s2id90.classification.data.digits.datasets;

import java.io.IOException;
import java.util.List;
import nl.tue.s2id90.classification.data.LabeledDataset2;
import nl.tue.s2id90.classification.data.digits.LabeledImage;
import nl.tue.s2id90.classification.data.digits.features.Doubles;

/**
 *
 * @author iVerb
 */
public class HandWrittenDatasetDoubles extends LabeledDataset2<Doubles, Byte> {

    public HandWrittenDatasetDoubles() {
    }
    
    public HandWrittenDatasetDoubles(List<LabeledImage> images) throws IOException {
        for(LabeledImage image : images) {
            Doubles features = new Doubles(image);
            put(image.getLabel(), features);
        }
    }

    @Override
    public LabeledDataset2<Doubles, Byte> createDataset() {
        return new HandWrittenDatasetDoubles();
    }
}
