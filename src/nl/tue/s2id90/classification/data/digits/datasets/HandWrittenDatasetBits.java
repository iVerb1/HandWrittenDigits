
package nl.tue.s2id90.classification.data.digits.datasets;

import java.io.IOException;
import java.util.List;
import nl.tue.s2id90.classification.data.LabeledDataset2;
import nl.tue.s2id90.classification.data.digits.LabeledImage;
import nl.tue.s2id90.classification.data.digits.features.Bits;

/**
 *
 * @author iVerb
 */
public class HandWrittenDatasetBits extends LabeledDataset2<Bits,Byte> {

    public HandWrittenDatasetBits() {
    }
    
    public HandWrittenDatasetBits(List<LabeledImage> images) throws IOException {
        for(LabeledImage image : images) {
            Bits features = new Bits(image);
            put(image.getLabel(), features);
        }
    }

    @Override
    public LabeledDataset2<Bits, Byte> createDataset() {
        return new HandWrittenDatasetBits();
    }
    
    
}
