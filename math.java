package StreamClustering.ReSSL;

import weka.core.Instance;

/**
 * Arimithetics computations
 * @author Administrator
 *
 */
public class math {
	
   public static double[] sub(double[] a, double[] b) {
      double[] r = new double[a.length];  
      for (int i=0; i<r.length; ++i) {
    	  r[i] = a[i]-b[i];
      }
      return r;
   }
   
   public static double[] sub(Instance a, double[] b) {
      double[] r = new double[b.length];  
      for (int i=0; i<r.length; ++i) {
    	  r[i] = a.value(i)-b[i];
      }
      return r;
   }
   
   public static double squareSum(double[] a) {
	   double r = 0;
	   for (int i=0; i<a.length; ++i) {
		   r += a[i]*a[i];
	   }
	   return r;
   }
   
   public static double euclideanDist(double[] a, double[] b) {
	   return Math.sqrt(squareSum(sub(a, b)));
   }
   
   public static double euclideanDist(Instance sa, Instance sb) {
		// TODO Auto-generated method stub
		double r = 0;
		for (int i=0; i<sa.numAttributes()-1; ++i) {
			r += Math.pow(sa.value(i)-sb.value(i), 2);
		}
		
		return Math.sqrt(r);
   }

   public static double min(double[] ds) {
	   // TODO Auto-generated method stub
	   double minV = ds[0];
	   for (int i=1; i<ds.length; ++i) {
		   if (ds[i] < minV) {
			   minV = ds[i];
		   }
	   }
	   return minV;
   }
   
   public static int minIndex(double[] v) {
	   int r = 0;
	   double minValue = v[0];
	   for (int i=1; i<v.length; ++i) {
		   if (v[i] < minValue) {
			   minValue = v[i];
			   r = i;
		   }
	   }
	   return r;
   }

   public static double sum(double[] minRVal) {
		// TODO Auto-generated method stub
		double sum = 0;
		for (int i=0; i<minRVal.length; ++i) {
			sum += minRVal[i];
		}
		return sum;
   }

	public static int[] minRVIndex(double[][] distM) {
		// TODO Auto-generated method stub
		int[] belongs = new int[distM.length];
		for (int i=0; i<distM.length; ++i) {
			belongs[i] = minIndex(distM[i]);
		}
		return belongs;
	}

	public static int maxIndex(int[] v) {
		// TODO Auto-generated method stub
		int maxIndex = 0;
		int maxValue = v[0];
		for (int i=1; i<v.length; ++i) {
			if (v[i] > maxValue) {
				maxIndex = i;
				maxValue = v[i];
			}
		}
		return maxIndex;
	}

	public static int sum(int[] v) {
		// TODO Auto-generated method stub
		int r = 0;
		for (int i=0; i<v.length; ++i) {
			r += v[i];
		}
		return r;
	}
	
	public static double[] normalize(double[] a) {
		double[] r = new double[a.length];
		double ret = sum(a);
		if (ret!=0) {
			for (int i=0; i<a.length; ++i) {
				r[i] = a[i] / ret;
			}
		}
		return r;
	}
	
	public static double[] normalize(int[] v) {
		double[] r = new double[v.length];
		double sumr = sum(v);
		for (int i=0; i<v.length; ++i) {
			r[i] = (double)r[i] / sumr;
		}
		return r;
	}
	
	public static double log2(double v) {
		return Math.log(v)/Math.log(2);
	}

	public static boolean belongs(int v, int[] lIndex) {
		// TODO Auto-generated method stub
		for (int i=0; i<lIndex.length; ++i) {
			if (lIndex[i] == v) {
				return true;
			}
		}
		
		return false;
	}

	public static int minPositive(int[] counts) {
		// TODO Auto-generated method stub
		int minP = 200;
		for (int i=0; i<counts.length; ++i) {
			if (counts[i]>0 && counts[i]<minP) {
				minP = counts[i];
			}
		}
		return minP;
	}
	
	public static double minPositive(double[] v) {
		// TODO Auto-generated method stub
		double minP = 200;
		for (int i=0; i<v.length; ++i) {
			if (v[i] > 0 && v[i] < minP) {
				minP = v[i];
			}
		}
		assert(minP!=200);
		return minP;
	}

	public static double euclideanDist(double[] ins, Instance instance) {
		double r = 0;// TODO Auto-generated method stub
		assert(ins.length==instance.numAttributes()-1);
		
		for (int i=0; i<ins.length; ++i) {
			r += Math.pow(ins[i], instance.value(i));
		}
		return Math.sqrt(r);
	}

	public static int maxIndex(double[] weights) {
		// TODO Auto-generated method stub
		int id = 0;
		double mw = weights[0];
		for (int i=1; i<weights.length; ++i) {
			if (weights[i] > mw) {
				id = i;
				mw = weights[i];
			}
		}
		return id;
	}

	public static double max(double[] weights) {
		// TODO Auto-generated method stub
		double r = weights[0];
		for (int i=1; i<weights.length; ++i) {
			if (weights[i] > r) {
				r = weights[i];
			}
		}
		return r;
	}

	public static double mean(double[] val) {
		// TODO Auto-generated method stub
		double ret = 0;
		for (int i=0; i<val.length; ++i) {
			ret += val[i];
		}
		return ret/val.length;
	}

	public static double std(double[] vals, double bias) {
		// TODO Auto-generated method stub
		double ret = 0;
		for (int i=0; i<vals.length; ++i) {
			ret += Math.pow((vals[i]-bias), 2);
		}
		return Math.sqrt(ret/vals.length);
	}
}
