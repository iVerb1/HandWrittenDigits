package nl.tue.s2id90.classification.data;

/**
 *
 * @author iVerb
 */
public class AttributeSplit {
    
    public final int index;
    public final Object value;
    public final boolean upperBound;
    public final boolean isContinuous;
    
    public AttributeSplit(int index, Number continuousValue, boolean upperBound) {
        this.index = index;
        this.value = continuousValue;
        this.upperBound = upperBound;
        this.isContinuous = true;
    }
    
    public AttributeSplit(int index, Object discreteValue) {
        this.index = index;
        this.value = discreteValue;
        this.isContinuous = false;
        this.upperBound = false;
    }
    
    @Override
    public String toString() {
        if (!isContinuous)
            return value.toString();
        else if (upperBound)
            return "<=" + value;
        else
            return ">" + value;
    }
    
}
