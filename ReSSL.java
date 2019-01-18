package StreamClustering.ReSSL;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

/**
 * Reliable Semi-supervised learning. Junming Shao, Chen Huang, Qinli Yang, Guangchun Luo. ICDM2016
 * @author masterqkk
 * @date: 2018/12/08
 *
 */
public class ReSSL {
	 protected static long t; 
     int kn;
     int tao;
     int kcluInit;
     int maxClu;
     
     int batchSize; // determine the size of samples used for batch clustering and interval for reporting accuracy 
     double labeledRate;
     double lambda;
     
     Instances structure;
     Instances datas;
     
     int numClass;
	 int dim;
	 
     int maxIter;
     
     ArrayList<semiClu> semiclusters;
     ArrayList<semiClu> outclus;
     
     double numL;
     double[] numLabeledEachClass;
     double[] distrl;
     double H;
     
     double[] accuracys;
     double averageAcc;
     
     public ReSSL(int kn, int tao, int kcluInit, int maxClu, double labeledRate, int batchSize, int maxIter, double lambda) {
    	 this.kn = kn;
    	 this.tao = tao;
    	 this.kcluInit = kcluInit;
    	 this.maxClu = maxClu;
    	 this.labeledRate = labeledRate;
    	 this.batchSize = batchSize;
    	 this.maxIter = maxIter;
    	 
    	 this.semiclusters = new ArrayList<semiClu>();
    	 this.outclus = new ArrayList<semiClu>(); 
    	 this.numL = 0;
    	 this.H = 0;
    	 this.lambda = lambda;
     }
     
     /**
      * ReSSL algorithm: dynamically maintain cluster structure online, including global and local data information.
      * @param dataPath
      * @param outPath
      * @param seed
      * @throws IOException
      */
     public void ExecuReSSL(String dataPath, String outPath, int seed) throws IOException {
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File(dataPath));
		structure =  loader.getStructure();
		datas = loader.getDataSet();
		datas.setClassIndex(datas.numAttributes()-1);
		structure.setClassIndex(datas.numAttributes()-1);
 		
		numClass = structure.numClasses();
		dim = structure.numAttributes()-1;
		
		int batchNum = datas.numInstances() / batchSize + (datas.numInstances() % batchSize == 0 ? 0:1);
		if (datas.numInstances()%batchSize==1) { // cope with spambase
			batchNum -= 1;
		}
		accuracys = new double[batchNum];
		
		numLabeledEachClass = new double[structure.numClasses()];
		distrl = new double[structure.numClasses()];
		boolean[] labelInfo = getLabeledInfo(batchSize, labeledRate, seed);
//		System.out.println(datas.instance(0));
    	// Initiial batch clustering 
		Instance[] initBatch = getBatch(0);
//		System.out.println(initBatch[0]);
		kmeans km = new kmeans(kcluInit, maxIter);
//		System.out.println("batch clustering start...");
        int[] belongs = km.doKmenas(initBatch);
	   	
        for (int i=0; i<kcluInit; ++i) {
        	semiClu scu = new semiClu(structure.numAttributes()-1, structure.numClasses());
        	scu.setWeight(1.0);
        	semiclusters.add(scu);
        }
        
        for (int i=0; i<initBatch.length; ++i) {
        	if(labelInfo[i]) {
        		numL++;
        		int tls = (int) initBatch[i].classValue();
        		numLabeledEachClass[tls]++;
        	}
        	int bl = belongs[i];
        	semiclusters.get(bl).insert(initBatch[i], labelInfo[i]);
        }
        updateH(); // update distrl and H
        
//        System.out.println("initial batch clustering finished.");
	    // online cluster structure maintainance
        double totalCount = 0, totalCorCount = 0;
        double tmpCount = 0, tmpCorCount = 0;
//        int t; // time stamp
        int ends = datas.numInstances();
        if (datas.numInstances()%batchSize==1) {
        	ends = datas.numInstances()-1;
        }
	    for (int i=batchSize; i<ends/*datas.numInstances()*/; ++i) {
//	    	t= i-batchSize;
	    	
	    	Instance inst = datas.instance(i);
	    	for (int icu=0; icu<semiclusters.size(); ++icu) {
	    		double tpDist = math.euclideanDist(semiclusters.get(icu).getCentroid(), inst);
	    		semiclusters.get(icu).setDist(tpDist);
	    	}
	    	Collections.sort(semiclusters);
	    	
	    	// compute reliability information
	    	double[] Rkn = new double[kn];
	    	for (int ik=0; ik<Math.min(kn, kcluInit); ++ik) {
	    		double CR; // CR(ik)
	    		if (semiclusters.get(ik).numL==0) {
	    			CR = H;
	    		}else {
	    			CR = (H-semiclusters.get(ik).Hci) / H;
	    		}
	    		double CP = 0; // CP(ik)
	    		for (int ic=0; ic<numClass; ++ic) {
	    			if (distrl[ic]!=0) {
	    				CP += (-1)*(semiclusters.get(ik).distrl[ic]-distrl[ic]) / distrl[ic];
	    			}
	    		}
	    		CP = 1/ Math.exp(CP);
	    		Rkn[ik] = CR * CP;
	    	}
	    	double bias = math.mean(Rkn);
	    	double std  = math.std(Rkn, bias);
	    	// predict
	    	double pl;
	    	if (Rkn[0] > bias + tao*std) {
	    		pl=0;
	    		double Dcs = math.euclideanDist(semiclusters.get(0).centroidEachClass[0], inst);
	    		double maxTar = Rkn[0]*semiclusters.get(0).distrl[0] / Dcs;
	    		for (int ic=1; ic<numClass; ++ic) {
	    			double tmpDcs = math.euclideanDist(semiclusters.get(0).centroidEachClass[ic], inst);
	    			double tmpTar = Rkn[ic]*semiclusters.get(0).distrl[ic] / tmpDcs;
	    			if (tmpDcs > maxTar) {
	    				pl = ic;
	    				maxTar = tmpTar;
	    			}
	    		}
	    	}else {
	    		pl=0;
	    		double maxTar = 0;
	    		for (int ik=0; ik<Math.min(kn, kcluInit); ++ik) {
	    			double Dcs = math.euclideanDist(semiclusters.get(ik).centroidEachClass[0], inst);
	    			maxTar += Rkn[ik] * semiclusters.get(ik).distrl[0]/ Dcs;
	    		}
	    		for (int ic=1; ic<numClass; ++ic) {
	    			double tmpTar = 0;
	    			for (int ik=0; ik<Math.min(kn, kcluInit); ++ik) {
	    				double Dcs = math.euclideanDist(semiclusters.get(ik).centroidEachClass[ic], inst);
	    				tmpTar += Rkn[ik] * semiclusters.get(ik).distrl[ic] / Dcs;
	    			}
	    			if (tmpTar > maxTar) {
	    				pl = ic;
	    				maxTar = tmpTar;
	    			}
	    		}
	    	}
	    	
	    	tmpCount++;
	    	if (pl == inst.value(dim)) {
	    		tmpCorCount++;
	    	}
	    	
	    	// output accuracy
	    	if (i%batchSize==batchSize-1 || i== datas.numInstances()-1) {
	    		totalCount += tmpCount;
	    		totalCorCount += tmpCorCount;
//	    		System.out.println(i);
	    		accuracys[i/batchSize] = totalCorCount/ totalCount;
	    		
	    		// retset
	    		tmpCorCount = 0; tmpCount = 0;
	    	}
	    	
	    	// update cluster structure
	    	double distn0 = semiclusters.get(0).dist;
	    	double radius0 = getRadius(semiclusters.get(0));
	    	double diamater0 = getDiamater(semiclusters.get(0));
	    	if (distn0 < radius0) {
	    		semiclusters.get(0).insert(inst, labelInfo[i%batchSize]);
	    		// update weights
	    		updateWeights();
	    		semiclusters.get(0).addWeight(1.0);		
	    	}else if (distn0 > diamater0) {
	    		if (outclus.size()==0) {
	    			semiClu outlierclu = new semiClu(inst, labelInfo[i%batchSize]);
	    			outclus.add(outlierclu);
	    		}else {
	    			if (outclus.size()>1) {
	    				for (int icu=0; icu<outclus.size(); ++icu) {
	    					double tdist = math.euclideanDist(outclus.get(icu).getCentroid(), inst);
		    				outclus.get(icu).setDist(tdist);
		    			}
	    				Collections.sort(outclus);
	    			}
	    			outclus.get(0).insert(inst, labelInfo[i%batchSize]);
	    		}
	    	}else {
	    		double radius1 = getRadius(semiclusters.get(1));
	    		double r = ((semiclusters.get(1).dist-radius1) - (semiclusters.get(0).dist-radius0)) / radius0;
	    		if (r>=0.5) {
	    			semiclusters.get(0).insert(inst, labelInfo[i%batchSize]);
	    			// update weights
		    		updateWeights();
		    		semiclusters.get(0).addWeight(1.0);
	    		}else {
	    			if (semiclusters.size() >= maxClu) {
	    				int minWIndex = 0;
	    				double minW = semiclusters.get(0).weight;
	    				for (int icu=0; icu<semiclusters.size(); ++icu) {
	    					double tW = semiclusters.get(icu).weight;
	    					if (tW < minW) {
	    						minW = tW;
	    						minWIndex = icu;
	    					}
	    				}
	    				semiclusters.remove(minWIndex);
	    			}
	    			// update weights
		    		updateWeights();
		    		
	    			semiClu newclu = new semiClu(inst, labelInfo[i%batchSize]);
	    			newclu.setWeight(1.0);
	    			semiclusters.add(newclu);
	    		}
	    	}	
	    	
	    	// update  global entropy et al. information 
	    	if (labelInfo[i%batchSize]) {
	    		numL++;
	    		numLabeledEachClass[(int) inst.classValue()]++;
	    		updateH(); // update distrl and H
	    	}
	    }	    
	    averageAcc = totalCorCount/ totalCount;
	 }
     
     
     private double getDiamater(semiClu semiClu) {
		// TODO Auto-generated method stub
    	double r=0;
    	double n = semiClu.numL + semiClu.numU;
        double[] LS = semiClu.LS;
        double inp = getInnerProduct(LS, LS);
        r = Math.sqrt((2*n*semiClu.SS-2*inp)/(n*(n-1)));
		return r;
	}

	private double getRadius(semiClu semiClu) {
		// TODO Auto-generated method stub
    	double n = semiClu.numL + semiClu.numU;
    	double[] ls = semiClu.LS;
    	double inp = getInnerProduct(ls, ls);
    	double r = Math.sqrt((n*semiClu.SS-inp)/(n*n));
		return r;
	}

	private double getInnerProduct(double[] ls, double[] ls2) {
		// TODO Auto-generated method stub
		double r = 0;
		for (int i=0; i<ls.length; ++i) {
			r += ls[i]*ls2[i];
		}
		return r;
	}

	private Instance[] getBatch(int bt) {
		// TODO Auto-generated method stub
		Instance[] batch = new Instance[batchSize];
		int base = bt*batchSize;
		int end = Math.min((bt+1)*batchSize, datas.numInstances());
		for (int i=base; i<end; ++i) {
			batch[i] = datas.instance(i);
		}
		return batch;
	 }
     
     private boolean[] getLabeledInfo(int chunkSize, double labeledRate, int seed) {
 		// TODO Auto-generated method stub
 		boolean[] info = new boolean[chunkSize];
 		ArrayList<Integer> totalIndex = new ArrayList<Integer>();
 		for (int i=0; i<chunkSize; ++i) {
 			totalIndex.add(new Integer(i));
 		}
 		int sed = seed;
 		Random random = new Random(sed);
 		
 		ArrayList<Integer> labelIndex = new ArrayList<Integer>();
 		int labeledCount = (int) (chunkSize * labeledRate);
 		for (int i=0; i<labeledCount; ++i) {
 			int next = totalIndex.remove(random.nextInt(totalIndex.size()));
 			//labelIndex.add(index);
 			info[next] = true;
 		}
// 		for (int i=0; i<labeledCount; ++i) {
// 			info[labelIndex.get(i)] = true;
// 		}
 		return info;
 	}    
     
     
    public void updateH() {
    	H = 0;
		for (int i=0; i<numClass; ++i) {
			distrl[i] = numLabeledEachClass[i]/numL;
			if (numLabeledEachClass[i] !=0) {
				H += -1 * (numLabeledEachClass[i]/numL) * Math.log(numLabeledEachClass[i]/numL);
			}
		}
    }
    
    // weight evolving with time
    public  void updateWeights() {
    	for (int i=0; i<semiclusters.size(); ++i) {
    		double w = semiclusters.get(i).weight;
    		semiclusters.get(i).setWeight(w*Math.exp(-lambda));
    	}
    }
}
