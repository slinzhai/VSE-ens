package config;

import model.VSEns;
import model.OptAUC;
import model.WARP;

/**
 * @author Weinan Zhang
 * 28 Feb. 2013
 */

public class Config {
	public static enum Datasets {OpenImages, NUS-WIDE, IAPR_TC12, Other}
	public static Datasets dataset = Datasets.IAPR_TC12;

	static{
		switch(dataset){
		case IAPR_TC12: // Your dataset
			imageNum = ;
			annoNum = ;
			K = 100;
			eta = 0.01;    // Learning rate
			maxRate = 1;
			lambdau = 0.01;
			lambdai = 0.01;
			initDelta = 0.01; //Initial upper value for p and q
			trainFile = ;
			testFile = ;
			splitChar = " ";
			roundNum = 10;
			showResultsEachRound = true;
			relevantThresholdScore = 0;
			break;
		case Other: // Your dataset
			break;
		}
	}

	public static int annoNum;
	public static int imageNum;
	public static int K;

	public static double eta;
	public static double lambdau;
	public static double lambdai;
	public static double initDelta;
	
	public static String trainFile;
	public static String testFile;
	public static String splitChar;
	
	public static int roundNum;
	public static boolean showResultsEachRound;
	public static long randomSeed = 10;
	public static int topNEvaluate = 5;
	public static double relevantThresholdScore;
	public static double maxRate;
	public static String annoPopFile;
	public static String twoCurveFigScriptFile;
	public static double meanStdevSmooth = 0.2;
	public static int maxImageNum = Integer.MAX_VALUE;//1000;
	public static double varianceLambda = 1; //0.1 0.01
	public static boolean varianceIsDownside = true;

	public static VSEns vsens = new VSEns();
	public static OptAUC optauc = new OptAUC();
	public static WARP warp = new WARP();

	public static double b = 20; //0.1;

}
