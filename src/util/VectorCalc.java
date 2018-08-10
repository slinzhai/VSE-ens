package util;

import java.util.Random;

import config.Config;

/**
 * @author Weinan Zhang
 * 28 Feb 2013
 */

public class VectorCalc {
	public static double euclidDistance1(double[] p, double[] q){
		double r = 0;
		for(int k = 0; k < p.length; ++k)
			r += Math.pow(Math.abs(p[k] - q[k]), 3);
		return Math.pow(r, 1.0/3);
	}
	
	public static double euclidDistance2(double[] p, double[] q){
		double r = 0;
		for(int k = 0; k < p.length; ++k)
			r += (p[k] - q[k]) * (p[k] - q[k]) ;
		return Math.sqrt(r);
	}
	
	public static double[] vectorMinus(double[] p, double[] q){
		double[] res = new double[p.length];
		for(int i = 0; i < p.length; ++i)
			res[i] = p[i] - q[i];
		return res;
	}
	public static double[] vectorPlus(double[] p, double[] q){
		double[] res = new double[p.length];
		for(int i = 0; i < p.length; ++i)
			res[i] = p[i] + q[i];
		return res;
	}
	public static double[] vectorDivide(double[] p, double d){
		double[] res = new double[p.length];
		for(int i = 0; i < p.length; ++i)
			res[i] = p[i] / d;
		return res;
	}
	public static double[] vectorTimes(double[] p, double d){
		double[] res = new double[p.length];
		for(int i = 0; i < p.length; ++i)
			res[i] = p[i] * d;
		return res;
	}
	
	public static double vectorInnerProduct(double[] p, double[] q){
		double r = 0;
		for(int k = 0; k < p.length; ++k)
			r += p[k] * q[k];
		return r;
	}
	public static double vectorInnerProductNormalized(double[] p, double[] q){
		double r = 0;
		double normp = 0;
		double normq = 0;
		
		for(int k = 0; k < p.length; ++k){
			r += p[k] * q[k];
			normp += p[k] * p[k];
			normq += q[k] * q[k];
		}
		return r / normp / normq;
	}
	
	public static double vectorInnerProduct(Double[] p, Double[] q){
		double r = 0;
		for(int k = 0; k < p.length; ++k)
			r += p[k] * q[k];
		return r;
	}
	
	public static double[] vectorElementProduct(double[] p, double[] q){
		double[] r;
		r = new double[p.length];
		for(int k = 0; k < p.length; ++k)
			r[k] = p[k] * q[k];
		return r;
	}
	
	public static double vectorMaxElement(double[] p){
		double max = Double.MIN_VALUE;
		for(int i = 0; i < p.length; i++)
		{
			if(p[i] > max)
				max = p[i];
		}
		return max;
	}
	
	public static double[] initCloseZeroVector(int K){
		Random rand = new Random();
		double[] res = new double[K];
		for(int i = 0; i < K; ++i)
			res[i] = (rand.nextDouble() - 0.5) * 2.0 * Config.initDelta;
		return res;
	}
	
	public static double[] initCloseWarmRandomVector(int K){
		Random rand = new Random();
		double[] res = new double[K];
		for(int i = 0; i < K; ++i)
			res[i] = (rand.nextDouble() - 0.5) * 2.0 * Config.initDelta;
		return res;
	}
	
	public static double[] matrixVectorProduct(double[][] m, double[] v){
		// return m * v
		if(m[0].length != v.length)
			System.err.println("matrixVectorProduct error: dimensino dismatch.");
		
		double[] res = new double[m.length];
		for(int i = 0; i < res.length; ++i){
			double sum = 0;
			for(int j = 0; j < m[i].length; ++j)
				sum += m[i][j] * v[j];
			res[i] = sum;
		}
		return res;
	}
	
	public static double[] matrixDiagonal(double[][] m){
		double[] res = new double[m.length];
		for(int i = 0; i < m.length; ++i)
			res[i] = m[i][i];
		return res;
	}
	public static double[] matrixDiagonalSqrt(double[][] m){
		double[] res = new double[m.length];
		for(int i = 0; i < m.length; ++i)
			res[i] = Math.sqrt(m[i][i]);
		return res;
	}
	
	public static void printMatrix(double[][] A, String name){
		System.out.println();
		System.out.println(name + ":");
		for(int i = 0; i < A.length; ++i){
			for(int j = 0; j < A[0].length; ++j)
				System.out.print(" " + A[i][j]);
			System.out.print("\n\n");
		}
	}
	
	public static double vectorLength(double[] q){
		double res = 0;
		for(double qi : q)
			res += qi * qi;
		return Math.sqrt(res);
	}
	
	public static double vectorCosine(double[] p, double[] q){
		return VectorCalc.vectorInnerProduct(p, q) / VectorCalc.vectorLength(p) / VectorCalc.vectorLength(q);
	}
	
	public static double vectorPearsonCorrelation(double[] p, double[] q){
		double[] pp = new double[p.length];
		double[] qq = new double[p.length];
		double pmean = vectorMeanValue(p);
		double qmean = vectorMeanValue(p);
		for(int i = 0; i < p.length; ++i){
			pp[i] = p[i] - pmean;
			qq[i] = q[i] - qmean;
		}
		return VectorCalc.vectorInnerProduct(pp, qq) / VectorCalc.vectorLength(pp) / VectorCalc.vectorLength(qq);
	}
	
	public static double vectorMeanValue(double[] p){
		double mean = 0;
		for(double pp : p)
			mean += pp;
		mean /= p.length;
		return mean;
	}
}
