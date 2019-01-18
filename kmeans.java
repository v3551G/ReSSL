package StreamClustering.ReSSL;

import java.util.ArrayList;
import java.util.Random;

import weka.core.Instance;
/**
 * Kmeans clustering with kmenas++ initialization
 * @author masterqkk
 *
 */
public class kmeans {
	int kcluInit;
	int maxIter;
	
	double eps = 0.05;
	
    public kmeans(int kcluInit, int maxIter) {
    	this.kcluInit = kcluInit;
    	this.maxIter = maxIter;
    }
    
    public int[] doKmenas(Instance[] batchori) {
    	ArrayList<double[]> batch = new ArrayList<double[]>();
    	int dim = batchori[0].numAttributes()-1;
    	for (int i=0; i<batchori.length; ++i) {
//    		Instance ins = (Instance) batchori[i].copy();
    		Instance ins = new Instance(batchori[i]);
    		ins.deleteAttributeAt(dim);
    		batch.add(ins.toDoubleArray());
    	}
		// used for KMeans++ initialization
    	int batchSize = batchori.length;
	   	 int seedc = 12345;
	   	 Random random =new Random(seedc);
	   	 double[][] means = new double[kcluInit][];
	   	 String indexs = "";
	   	 // select seeds
	   	 int itIndex = random.nextInt(batchSize);
//	   	 batch[itIndex].deleteAttributeAt(dim);
	   	 means[0] = (double[]) batch.get(itIndex);
	   	 indexs += itIndex + ",";
	   	 for (int ii=1; ii<kcluInit; ++ii) {
	   		 double[][] distM = new double[batchSize][ii];
	   		 for (int j=0; j<batchSize; ++j) {
	   			 for (int ki=0; ki<ii; ++ki) {
	   				 distM[j][ki] = math.squareSum( math.sub(batch.get(j), means[ki]) );
	   			 }
	   		 }
	   		 int index = Roulettemethod(distM);
//	   		 batch[index].deleteAttributeAt(dim);
	   		 means[ii] = (double[]) batch.get(index); // ? 
	   		 indexs += index + ",";
	   	 }
	   
	     // Kmenas clustering
	   	 int[] belongs = new int[batchSize];
	   	
	   	 double target;
	   	 int iter = 0;
	   	 do {
	   	    target = 0;
	   		// compute belongs
	   		for (int ins=0; ins<batchSize; ++ins) {
	   			double dist = math.euclideanDist( means[0], batch.get(ins) );
	   			int minId = 0;
	   			for (int clu=1; clu<kcluInit; clu++) {
	   				double tmpDist = math.euclideanDist( means[clu], batch.get(ins));
	   				if (tmpDist < dist) {
	   					dist = tmpDist;
	   					minId = clu;
	   				}
	   			}
	   			belongs[ins] = minId;
//	   			target += math.euclideanDist( means[minId], batch[ins]);
	   		}
	   		// update centroids
	   		double[][] newMeans = new double[kcluInit][dim];
	   		for (int ins=0; ins<batchSize; ++ins) {
	   			int bl = belongs[ins];
	   			for (int di=0; di<dim; ++di) {
	   				newMeans[bl][di] += batch.get(ins)[di];
	   			}
	   		}
	   		for (int i=0; i<kcluInit; ++i) {
	   			target += math.euclideanDist(means[i], newMeans[i]);
	   		}
	   		means = newMeans.clone(); 
	   		iter++;
	   	 }while(iter<maxIter && target>eps);
	   	 
	   	 return belongs;
    }
    
    /**
     * get next center index by Roulette method
     * @param distM
     * @return
     */
    private int Roulettemethod(double[][] distM) {
		// TODO Auto-generated method stub
	   	int r = 0;
	   	double[] minRVal = new double[distM.length]; 
	   	for (int i=0; i<minRVal.length; ++i) {
	   		minRVal[i] = math.min(distM[i]);
	   	}
	   	double sum_mV = math.sum(minRVal);
	   	for (int i=0; i<minRVal.length; ++i) {
	   		minRVal[i] = minRVal[i] / sum_mV;
	   	}
	   	double[] temp_roulette = new double[minRVal.length];
	   	temp_roulette[0] = minRVal[0];
	   	for (int i=1; i<temp_roulette.length; ++i) {
	   		temp_roulette[i] = temp_roulette[i-1] + minRVal[i];
	   	}
	   	Random random = new Random(1234);
	   	double thread = random.nextDouble(); // ??
	   	for (int i=0; i<temp_roulette.length; ++i) {
	   		if (i==0 && temp_roulette[i] > thread) {
	   			r =  1;
	   		}else if (temp_roulette[i] > thread && temp_roulette[i-1] < thread){
	   			r = i;
	   		}
	   	}
	   	
	    return r;
	}
}
