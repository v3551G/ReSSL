package StreamClustering.ReSSL;

import sun.util.logging.resources.logging;
import weka.core.Instance;

/**
 * definition of semi-supervised cluster
 * @author masterqkk
 *
 */
public class semiClu implements Comparable<semiClu> {
    double[] LS;
    double SS;
    double numL;
    double numU;
    double[] numLabeledEachClass;
    double[][] centroidEachClass;
    
    double[] distrl;
    double Hci; //entropy on numLabeledEachClass
    
    int dim;
    int numClass;
    
    double dist; 
    double weight;
    
    public semiClu(int dim, int numClass) {
    	this.dim = dim;
    	this.numClass = numClass;
    	LS = new double[dim];
    	SS =0;
    	numL = 0;
    	numU = 0;
    	numLabeledEachClass = new double[numClass];
    	centroidEachClass = new double[numClass][dim];
    	
    	distrl = new double[numClass];
    	Hci = 0;
    	dist=0;
    }
    
    public semiClu(int numClass, double[] inss, int trueClass, boolean labeled) {
    	this.dim = inss.length;
    	this.numClass = numClass;
    	LS = new double[dim];
    	SS = 0;
    	for (int i=0; i<dim; ++i) {
    		LS[i] = inss[i];
    		SS += inss[i] * inss[i];
    	}
    	numL = 0;
    	numU = 0;
    	numLabeledEachClass = new double[numClass];
    	centroidEachClass = new double[numClass][dim];
    	if (labeled) {
    		numL++;
    		numLabeledEachClass[trueClass]++;
    		for (int i=0; i<dim; ++i) {
    			centroidEachClass[trueClass][i] = inss[i];
    		}
    	}else {
    		numU++;
    	}
    	
    	distrl = new double[numClass];
    	//TODO:
    	updateHCI();
    }
    
    public semiClu(Instance inss,  boolean labeled) {
    	this.dim = inss.numAttributes()-1;
    	this.numClass = inss.numClasses();
    	LS = new double[dim];
    	SS = 0;
    	for (int i=0; i<dim; ++i) {
    		LS[i] = inss.value(i);
    		SS += inss.value(i) * inss.value(i);
    	}
    	numL = 0;
    	numU = 0;
    	numLabeledEachClass = new double[numClass];
    	centroidEachClass = new double[numClass][dim];
    	if (labeled) {
    		numL++;
    		int tcls = (int) inss.value(dim);
    		numLabeledEachClass[tcls]++;
    		for (int i=0; i<dim; ++i) {
    			centroidEachClass[tcls][i] = inss.value(i);
    		}
    	}else {
    		numU++;
    	}
    	
    	distrl = new double[numClass];
    	//TODO: !!!!The computation is related with original dataset.
    	updateHCI();
    }
    
    // 
    public void insert(double[] inss, int trueClass, boolean labeled) {
    	for (int i=0; i<dim; ++i) {
    		LS[i] += inss[i];
    		SS += inss[i] * inss[i];
    	}
    
    	if (labeled) {
    		numL++;
    		numLabeledEachClass[trueClass]++;
    		for (int i=0; i<dim; ++i) {
    			centroidEachClass[trueClass][i] += inss[i];
    		}
    	}else {
    		numU++;
    	}
    	
    	//TODO: 
    	updateHCI();
    	
    }
    
    public void insert(Instance inss, boolean labeled) {
    	for (int i=0; i<dim; ++i) {
    		LS[i] += inss.value(i);
    		SS += inss.value(i) * inss.value(i);
    	}
    
    	if (labeled) {
    		numL++;
    		int tls = (int) inss.value(dim);
    		numLabeledEachClass[tls]++;
    		for (int i=0; i<dim; ++i) {
    			centroidEachClass[tls][i] += inss.value(i);
    		}
    	}else {
    		numU++;
    	}
    	
    	//TODO: 
    	updateHCI();
    }
    
    // update entropy and class distribution
    public void updateHCI() {
    	if (numL<1) {
			Hci=0;
		}else {
			Hci = 0;
			for (int i=0; i<numClass; ++i) {
				distrl[i] = numLabeledEachClass[i]/numL;
				if (numLabeledEachClass[i] !=0) {
					Hci += -1 * (numLabeledEachClass[i]/numL) * Math.log(numLabeledEachClass[i]/numL);
				}
			}
			
		}
    }
    
    // increasing order
	@Override
	public int compareTo(semiClu o) {
		// TODO Auto-generated method stub
		double val = this.dist-o.dist;
		if (val>0)
			return 1;
		else if (val<0) {
			return -1;
		}else
			return 0;
	}
	
	public void setDist(double dist) {
		this.dist = dist;
	}
    
	public double[] getCentroid() {
		double[] ret = new double[dim];
		for (int i=0; i<dim; ++i) {
			ret[i] = LS[i]/(numL+numU);
		}
		return ret;
	}

	public void setWeight(double weight) {
		// TODO Auto-generated method stub
		this.weight = weight; 
	}
	
	public void addWeight(double adw) {
		this.weight += adw;
	}
}
