package util;

import java.util.Random;

import config.Config;

/**
 * @author Weinan Zhang
 * 28 Feb. 2013
 */

public class Calc {
	public static Random rand = new Random(Config.randomSeed);
	
	public static double norm2(double q[]){
		double n = 0;
		for(int i = 0; i < q.length; ++i)
			n += q[i] * q[i];
		return Math.sqrt(n);
	}
	
	public static double sigmoid(double x){
		return 1.0 / (1.0 + Math.exp(- x));
	}
	
	public static double sigmoidToNegPos1(double x){
		return 2.0 * (1.0 / (1.0 + Math.exp(- x)) - 0.5);
	}
	
	public static double inverseSigmoidToNegPos1(double y){
		return - Math.log(2.0 / (y * 0.99 + 1.0) - 1);
	}
}
