package nl.tue.s2id90.classification.decisiontree;

/**
 *
 * @author huub
 */
public abstract class Information {
    
    /**
     * @param p probability distribution
     * @return the entropy of the probability distribution p. **/
    public static double entropy(double ... p) {
        double entropy = 0;
        for (double prob : p) {
            if (prob != 0)
                entropy += xlog2x(prob);
        }
        return - entropy;
    }
    
    /**
     * @param x argument
     * @return  2log(x)
     **/
    public static double log2(double x) {
        return (double) (Math.log(x) / Math.log(2));
    }
    
    /**
     * @param x argument
     * @return  x * 2log(x)
     **/
    public static double xlog2x(double x) {
        return (double) x * log2(x);
    }
}
