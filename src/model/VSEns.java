package model;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Hashtable;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.Map.Entry;

import config.Config;
import dataformat.ImageAnnoPairUnit;
import util.Discrete;
import util.Evaluation;
import util.Statistic;
import util.VectorCalc;
import util.TwoTuple;
import util.Lists;


/**
 * 
 * VSE-ens: Visual Semantic Embeddings with Efficient Negative Sampling
 * 
 * Guibing Guo, Songlin Zhai, Fajie Yuan
 * 
 * 
 **/

public class VSEns {
	public double[][] P;
	public double[][] Q;

	Hashtable<Integer, HashSet<Integer>> trainMatrix;
	Hashtable<Integer, HashSet<Integer>> testMatrix;
	public LinkedList<ImageAnnoPairUnit> testImageAnnoPairs;
	
	/**
	 * Parameters of VSE-ens model
	 **/
	double lambda = 0.001;
	private int updatingNum;
	private int lambdaNum;
	private double [] rankingPro;
	private double[] var;
	private int[][] factorRanking;

	public void driver() throws IOException{
		System.out.println("Loading data..........");
		this.initTrainMatrix();
		this.initTestMatrix();
		this.readInTestData(Config.testFile);
		System.out.println("Loading data done.");
		this.initialize();
		System.out.println("Initializing parameters done.");
		System.out.println("\nStart training VSE-ens model...\n");
		this.train();
		System.out.println("\nFinish training VSE-ens model...");
		this.eval();
	}
	
	TwoTuple<Double, Integer> trainOneRound(int poscounter){
        Random rand = new Random(Config.randomSeed);

		Set<Integer> images_ = this.trainMatrix.keySet();
		Integer num_images = images_.size();
		Integer[] images = new Integer[num_images];
		images = images_.toArray(images);
		double loss = 0;
		int posCounter = poscounter;
		for(Integer p_ = 0; p_ < num_images*100; ++p_){
			if(posCounter % this.updatingNum == 0){
				// Update annotation ranking for every |I|log|I|
				updateRankingInFactor();
				posCounter = 0;
			}
			++posCounter;
			
			// Randomly sample an image
			Integer pid = images[rand.nextInt(num_images)];
			HashSet<Integer> posAnnos_ = this.trainMatrix.get(pid);
			Integer num_pos = posAnnos_.size();
			Integer [] posAnnos = new Integer[num_pos];
			posAnnos = posAnnos_.toArray(posAnnos);
			// Randomly sample positive annotation
			Integer iid = posAnnos[rand.nextInt(num_pos)];
			
			Integer jid = iid;
			while(this.trainMatrix.get(pid).contains(jid) || jid == 0){
				// Draw a r by Exp(-r/lambda)
				int randomJIndex = 0;
				do {
					Discrete d = new Discrete(this.rankingPro);
					randomJIndex = d.random();
				} while (randomJIndex > Config.annoNum);
				// Draw a f by p(f|c)
				double[] pfc = new double[Config.K];
				double sumfc = 0;
				for (int index = 0; index < Config.K; index++) {
					double temp = Math.abs(P[pid][index]);
					sumfc += temp * var[index];
					pfc[index] = temp * var[index];
				}
				for (int index = 0; index < Config.K; index++) {
					pfc[index] /= sumfc;
				}
				Discrete d = new Discrete(pfc);
				int f = d.random();
				// Select r-1 or r negative annotation in f
				jid = rand.nextInt(Config.annoNum);// Initialize a negative annotation randomly
				if (P[pid][f] > 0) {
					jid = factorRanking[f][randomJIndex];
				} else {
					jid = factorRanking[f][Config.annoNum - randomJIndex - 1];
				}
			}
			
			double hat_r_ui = VectorCalc.vectorInnerProduct(P[pid], Q[iid]);
			double hat_r_uj = VectorCalc.vectorInnerProduct(P[pid], Q[jid]);
			loss += this.getLoss(hat_r_ui - hat_r_uj);
			double gradient = this.getGradient(hat_r_ui - hat_r_uj);
			for(int k = 0; k < Config.K; ++k){
				P[pid][k] = P[pid][k] + Config.eta * (gradient * (Q[iid][k] - Q[jid][k]) - Config.lambdau * P[pid][k]);
				Q[iid][k] = Q[iid][k] + Config.eta * (gradient * P[pid][k] - Config.lambdai * Q[iid][k]);
				Q[jid][k] = Q[jid][k] + Config.eta * (- gradient * P[pid][k] - Config.lambdai * Q[jid][k]);

				loss += Config.lambdau*P[pid][k]*P[pid][k] + Config.lambdai*Q[iid][k]*Q[iid][k] + Config.lambdai*Q[jid][k]*Q[jid][k];
			}
		}
	    TwoTuple<Double, Integer> results = new TwoTuple<Double, Integer>(loss, posCounter);
		return results;
    }

	void train(){
		double auc = 0;
		double maxauc = Double.MIN_VALUE;
		
		double lastloss = 0;
		int poscounter = 0;
		
		for(int i = 0; i < Config.roundNum; ++i){
			TwoTuple<Double, Integer> result = trainOneRound(poscounter);
			double loss = result.first;
			poscounter = result.second;
			auc = testAUC();
			maxauc = Math.max(maxauc, auc);
			double deloss = loss - lastloss;
			if(Config.showResultsEachRound){
				System.out.println("Round " + (i+1) + "     AUC: " + auc + "     Loss: " + loss + "     DeltaLoss: " + deloss);
			}else{
				System.out.println("Round " + (i+1) + "     Loss: " + loss + "     DeltaLoss: " + deloss);
			}
			lastloss = loss;
		}
		auc = testAUC();
		maxauc = Math.max(maxauc, auc);
		System.out.println("\n< Optimal AUC: " + maxauc + " >");
	}

	double testAUC(){
		double auc = 0;
		for(ImageAnnoPairUnit uipu : testImageAnnoPairs){
			double r_ui = VectorCalc.vectorInnerProduct(P[uipu.pid], Q[uipu.iid]);
			double r_uj = VectorCalc.vectorInnerProduct(P[uipu.pid], Q[uipu.jid]);
			
			if(r_ui > r_uj)
				auc += 1;
		}
		auc /= testImageAnnoPairs.size();
		return auc;
	}
	
    void eval()throws IOException{
		System.out.println("\nEvaluating VSE-ens model..........\n");
		Evaluation evaluator = new Evaluation(this.trainMatrix,this.testMatrix,this.recommendRank());
		System.out.println("Precision@" + Config.topNEvaluate + ":  " + evaluator.PrecisionEvaluator());
		System.out.println("Recall@" + Config.topNEvaluate + ":  " + evaluator.RecallEvaluator());
		System.out.println("\nEnd evaluation..........");
	}

	double getGradient(double hat_r_ij){
		double gradient = 0;
		// Hinge loss
		if (hat_r_ij <= 1)
			gradient = 1;
		return gradient;
	}
	
	double getLoss(double hat_r_ij){
		// Hinge loss
		double loss = 0;
		if (hat_r_ij < 1)
			loss = -hat_r_ij + 1;
		return loss;
	}

	void readInTestData(String testFile) throws IOException{
		testImageAnnoPairs = new LinkedList<ImageAnnoPairUnit>();
		BufferedReader reader;
		String[] splits;
		reader = new BufferedReader(new FileReader(testFile));
		Random rand = new Random(Config.randomSeed);
		while(reader.ready()){
			splits = reader.readLine().split(Config.splitChar);
			int pid = Integer.parseInt(splits[0]);
			int iid = Integer.parseInt(splits[1]);
			float rate = Float.parseFloat(splits[2]);
			if(rate > Config.relevantThresholdScore && trainMatrix.containsKey(pid)){ // only images in train will be in test
				int jid = rand.nextInt(Config.annoNum);
				while(trainMatrix.get(pid) != null && trainMatrix.get(pid).contains(jid))
					jid = rand.nextInt(Config.annoNum);
				
				testImageAnnoPairs.add(new ImageAnnoPairUnit(pid, iid, jid));
			}
		}
		reader.close();
	}
	
	void initTrainMatrix()throws IOException{
		this.trainMatrix = new Hashtable<Integer, HashSet<Integer>>();
		BufferedReader reader;
		String[] splits;
		reader = new BufferedReader(new FileReader(Config.trainFile));
		while(reader.ready()){
			splits = reader.readLine().split(Config.splitChar);
			int pid = Integer.parseInt(splits[0]);
			int iid = Integer.parseInt(splits[1]);
			float rate = Float.parseFloat(splits[2]);
			if(rate > Config.relevantThresholdScore){
				if(!this.trainMatrix.containsKey(pid))
					this.trainMatrix.put(pid, new HashSet<Integer>());
				this.trainMatrix.get(pid).add(iid);
			}
		}
		reader.close();
	}
	
	void initTestMatrix()throws IOException{
		this.testMatrix = new Hashtable<Integer, HashSet<Integer>>();
		BufferedReader reader;
		String[] splits;
		reader = new BufferedReader(new FileReader(Config.testFile));
		while(reader.ready()){
			splits = reader.readLine().split(Config.splitChar);
			int pid = Integer.parseInt(splits[0]);
			int iid = Integer.parseInt(splits[1]);
			float rate = Float.parseFloat(splits[2]);
			if(rate > Config.relevantThresholdScore){
				if(!this.testMatrix.containsKey(pid))
					this.testMatrix.put(pid, new HashSet<Integer>());
				this.testMatrix.get(pid).add(iid);
			}
		}
		reader.close();
		HashSet<Integer> removeList = new HashSet<Integer>();
		for(Integer pid : testMatrix.keySet()){
			if (!this.trainMatrix.containsKey(pid)){removeList.add(pid);continue;} // If the current image not in train dataset
			System.out.print(pid);
			HashSet<Integer> images = this.testMatrix.get(pid);
			for (Integer iid : images){
				if (this.trainMatrix.get(pid).contains(iid)) this.testMatrix.remove(pid, iid);
			}
			if (this.testMatrix.get(pid) == null) removeList.add(pid);
		}
		for (Integer rid: removeList){
			this.testMatrix.remove(rid);
		}
	}
	
	// You can evaluate images one by one for optimization instead of ranking all once.
	Integer [][] recommendRank(){
		Integer [][] recommendedList = new Integer[Config.imageNum][Config.annoNum];
		for (Integer pid : testMatrix.keySet()){
			double [] ratings = new double[Config.annoNum];
			for(Integer iid = 0; iid < Config.annoNum; ++iid){
				double r_pi = VectorCalc.vectorInnerProduct(P[pid], Q[iid]);
				ratings[iid] = r_pi;
			}
			// Get the recommended list for current image
			List<Entry<Integer, Double>> result = this.sorteVectorValue(ratings, true);
			for(Integer index = 0; index < Config.annoNum; ++index){
				recommendedList[pid][index] = result.get(index).getKey();
			}
		}
		return recommendedList;
	}

	/**
	 * Utilities for updating the rankingPro[k][numAnno]
	 **/
	public void updateRankingInFactor() {
		// Rank the annotations by a specific f
		for (int factorIndex = 0; factorIndex < Config.K; factorIndex++) {
			double [] factorVector = this.getColumn(Q, factorIndex);
			List<Entry<Integer, Double>> sort = this.sorteVectorValue(factorVector, true);
			double[] valueList = new double[Config.annoNum];
			for (int i = 0; i < Config.annoNum; i++) {
				factorRanking[factorIndex][i] = sort.get(i).getKey();
				valueList[i] = sort.get(i).getValue();
			}
			Statistic s = new Statistic(valueList);
			var[factorIndex] = s.getVariance();
		}
	}

	public List<Entry<Integer, Double>> sorteVectorValue(double[] vector, boolean decending) {
		Map<Integer, Double> keyValPair = new HashMap<>();
		for (int i = 0; i < vector.length; i++) {
			keyValPair.put(i, vector[i]);
		}
		return Lists.sortMap(keyValPair, decending);
	}
	
	
	double [] getColumn(double[][] matrix, int column){
		double [] result = new double[Config.annoNum];
		for (int i = 0; i < Config.annoNum; ++i) result[i] = matrix[i][column];
		return result;
	}


	/**
	 * Initialize our model
	 **/
	void initialize() {
		Random rand = new Random(10);
		
		P = new double[Config.imageNum][Config.K];
		for(int u = 0; u < Config.imageNum; ++u){
			P[u] = new double[Config.K];
			for(int k = 0; k < Config.K; ++k)
				P[u][k] = (rand.nextDouble() - 0.5) * 2.0 * Config.initDelta; // [-initDelta, initDelta]
		}
		
		Q = new double[Config.annoNum][Config.K];
		for(int i = 0; i < Config.annoNum; ++i){
			Q[i] = new double[Config.K];
			for(int k = 0; k < Config.K; ++k)
				Q[i][k] = (rand.nextDouble() - 0.5) * 2.0 * Config.initDelta; // [-initDelta, initDelta]
		}
		
		// Set the \lambda * annoNum for the sampling the rank
		this.lambdaNum = (int) (lambda * (Config.annoNum-1));
		// Set the number of updating the ranking
		this.updatingNum = (int) ((Config.annoNum-1) * Math.log(Config.annoNum-1));
		// Store the variance of annotations latent factor
		var = new double[Config.K];
		
		this.lambda = 0.001;

		factorRanking = new int[Config.K][Config.annoNum];
		this.rankingPro = new double[Config.annoNum];
		// Set the ranking probility for sampling the rank
		double sum = 0;
		for (int i = 1; i < Config.annoNum; i++) {
			rankingPro[i] = Math.exp(-(i + 1) / lambdaNum);
			sum += rankingPro[i];
		}
		for (int i = 1; i < Config.annoNum; i++) {
			rankingPro[i] /= sum;
		}
	}

	public static void main(String[] args) throws IOException {
		Config.roundNum = 0;
		Config.showResultsEachRound = false;
		Config.eta = 0.01; // Learning rate
		Config.lambdau = 0.1; // Regulation for image
		Config.lambdai = 0.1; // Regulation for annotation
		Config.topNEvaluate = 5;
		Config.K = 100;
		Config.vsens.driver();
	}
}
