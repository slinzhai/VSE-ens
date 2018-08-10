package util;

import java.util.HashSet;
import java.util.Hashtable;

import config.Config;

/**
 * @author Songlin Zhai
 * 27 May 2018
 */

public class Evaluation {
	private Hashtable<Integer, HashSet<Integer>> testMatrix;
	private Hashtable<Integer, HashSet<Integer>> trainMatrix;
	private Integer [][] recommendedList;

	public Evaluation(Hashtable<Integer, HashSet<Integer>> trainmatrix, Hashtable<Integer, HashSet<Integer>> testmatrix, Integer[][] remdlist){
		this.trainMatrix = trainmatrix;
		this.testMatrix = testmatrix;
		this.recommendedList = remdlist;
	}
	
	public double PrecisionEvaluator (){
		double totalPrecision = 0.0;
		int nonZeroNumImages = 0;
        for (Integer imageID : testMatrix.keySet()) {
        	if (testMatrix.get(imageID) == null || this.trainMatrix.get(imageID) == null) continue;
            Integer[] recommendListByImage = this.recommendedList[imageID];
            int numHits = 0;
            int topK = Config.topNEvaluate <= recommendListByImage.length ? Config.topNEvaluate : recommendListByImage.length;

            int predanno = 0;
            for (int indexOfAnno = 0; indexOfAnno < recommendListByImage.length && predanno < topK; ++indexOfAnno) {
                Integer annoID = recommendListByImage[indexOfAnno];
                if(this.trainMatrix.get(imageID) != null){
                	if(this.trainMatrix.get(imageID).contains(annoID) || annoID == 0) continue;
                }
                ++predanno;
                if (testMatrix.get(imageID).contains(annoID)) numHits++;
            }
            totalPrecision += numHits / (predanno + 0.0);
            nonZeroNumImages++;
        }
        return nonZeroNumImages > 0 ? totalPrecision / nonZeroNumImages : 0.0d;
    }

	public double RecallEvaluator(){
		double totalRecall = 0.0;
		int nonZeroNumImages = 0;
        for (Integer imageID : testMatrix.keySet()) {
        	if (testMatrix.get(imageID) == null || this.trainMatrix.get(imageID) == null) continue;
        	Integer[] recommendListByImage = this.recommendedList[imageID];
            int numHits = 0;
            int topK = Config.topNEvaluate <= recommendListByImage.length ? Config.topNEvaluate : recommendListByImage.length;
            
            int predanno = 0;
            for (int i = 0; i < recommendListByImage.length && predanno < topK; ++i) {
                int annoID = recommendListByImage[i];
                if(this.trainMatrix.get(imageID) != null){
                	if(this.trainMatrix.get(imageID).contains(annoID) || annoID == 0) continue;
                }
                ++predanno;
                if (testMatrix.get(imageID).contains(annoID)) numHits++; 
            }
            totalRecall +=  numHits / (this.testMatrix.get(imageID).size() + 0.0);
            nonZeroNumImages++;
        }
        return nonZeroNumImages > 0 ? totalRecall / nonZeroNumImages : 0.0d;
	}

	public double MAPEvaluator(){
		double totalPrecision = 0.0;
        int nonZeroNumImages = 0;
        for (Integer imageID : testMatrix.keySet()) {
        	if (testMatrix.get(imageID) == null || this.trainMatrix.get(imageID) == null) continue;
          	Integer[] recommendListByImage = this.recommendedList[imageID];
            int numHits = 0;
            int topK = Config.topNEvaluate <= recommendListByImage.length ? Config.topNEvaluate : recommendListByImage.length;
            double tempPrecision = 0.0d;
            
            int predanno = 0;
            for (int indexOfItem = 0; indexOfItem < recommendListByImage.length && predanno < topK; ++indexOfItem) {
            	int annoID = recommendListByImage[indexOfItem];
            	if(this.trainMatrix.get(imageID) != null){
                	if(this.trainMatrix.get(imageID).contains(annoID) || annoID == 0) continue;
                }
            	++predanno;
             	if (testMatrix.get(imageID).contains(annoID)) {
                    numHits++;
                    tempPrecision += 1.0 * numHits / (indexOfItem + 1);
                }
            }
            totalPrecision += tempPrecision / (testMatrix.get(imageID).size() < topK ? testMatrix.get(imageID).size(): topK);
            nonZeroNumImages++;
        }
        return nonZeroNumImages > 0 ? totalPrecision / nonZeroNumImages : 0.0d;
	}
}
