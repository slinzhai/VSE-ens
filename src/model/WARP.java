package model;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
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
import util.Evaluation;
import util.VectorCalc;

/*
 * Revised by Songlin Zhai
 * 5 Aug 2018
 */

public class WARP {
	public double[][] P;
	public double[][] Q;

	private Hashtable<Integer, HashSet<Integer>> trainMatrix;
	private Hashtable<Integer, HashSet<Integer>> testMatrix;
	
	public LinkedList<ImageAnnoPairUnit> testImageAnnoPairs;


	public void driver() throws IOException{
		System.out.println("Loading data..........");
		this.initTrainMatrix();
		this.initTestMatrix();
		this.readInTestData(Config.testFile);
		System.out.println("Loading data done.");
		this.initialize();
		System.out.println("Initializing parameters done.");
		System.out.println("\nStart training WARP model...\n");
		this.train();
		System.out.println("\nFinish training WARP model...");
		this.eval();
	}

	
	void train() throws IOException{
		double auc = 0;
		double maxauc = Double.MIN_VALUE;
		
		for(int i = 0; i < Config.roundNum; ++i){
			trainOneRound();
			auc = calAUC();
			maxauc = Math.max(maxauc, auc);
			if(Config.showResultsEachRound){
				System.out.println("Round " + (i+1) + "\tAUC: " + auc);
			}else{
				System.out.println("Round " + (i+1) + " for training WARP model.");
			}
		}
		auc = calAUC();
		maxauc = Math.max(maxauc, auc);
		System.out.println("\n< Optimal AUC: " + maxauc + " >");
	}

	double calAUC(){
		double auc = 0;
		for(ImageAnnoPairUnit pipu : testImageAnnoPairs){
			double r_pi = VectorCalc.vectorInnerProduct(P[pipu.pid], Q[pipu.iid]);
			double r_pj = VectorCalc.vectorInnerProduct(P[pipu.pid], Q[pipu.jid]);

			if(r_pi > r_pj)
				auc += 1;
		}
		auc /= testImageAnnoPairs.size();
		return auc;
	}

	// You can evaluate images one by one for optimization instead of ranking all once.
	Integer [][] predictRank(){
		Integer [][] predictedList = new Integer[Config.imageNum][Config.annoNum];
		for (Integer pid : testMatrix.keySet()){
			double [] ratings = new double[Config.annoNum];
			for(Integer iid = 0; iid < Config.annoNum; ++iid){
				double r_pi = VectorCalc.vectorInnerProduct(P[pid], Q[iid]);
				ratings[iid] = r_pi;
			}
			// Get the recommended list for current image
			List<Entry<Integer, Double>> result = this.sortVectorDecend(ratings);
			for(Integer index = 0; index < Config.annoNum; ++index){
				predictedList[pid][index] = result.get(index).getKey();
			}
		}
		return predictedList;
	}

	
	void eval()throws IOException{
		System.out.println("\nEvaluating WARP model..........\n");
		Evaluation evaluator = new Evaluation(this.trainMatrix,this.testMatrix,this.predictRank());
		System.out.println("Precision@" + Config.topNEvaluate + ":  " + evaluator.PrecisionEvaluator());
		System.out.println("Recall@" + Config.topNEvaluate + ":  " + evaluator.RecallEvaluator());
		System.out.println("\nEnd evaluation..........");
	}


	public List<Entry<Integer, Double>> sortVectorDecend(double[] vector) {
		Map<Integer, Double> keyValPair = new HashMap<>();
		for (int i = 0, length = vector.length; i < length; i++) {
			keyValPair.put(i, vector[i]);
		}
		List<Entry<Integer, Double>> list = new ArrayList<>(keyValPair.entrySet());
		Collections.sort(list,new Comparator<Map.Entry<Integer, Double>>() {
            @Override
            public int compare(Map.Entry<Integer, Double> o1, Map.Entry<Integer, Double> o2) {
                return (o2.getValue()).compareTo(o1.getValue());
            }
        });
		return list;
	}


	double trainOneRound() throws IOException{
		Random rand = new Random(Config.randomSeed);
		double loss = 0;
		Set<Integer> images_ = this.trainMatrix.keySet();
		Integer num_images = images_.size();
		Integer[] images = new Integer[num_images];
		images = images_.toArray(images);
		for(Integer p_ = 0; p_ < num_images*100; ++p_){
			// Randomly sample an image
			Integer pid = images[rand.nextInt(num_images)];
			// Randomly sample a positive annotation
			HashSet<Integer> posAnnos_ = this.trainMatrix.get(pid);
			Integer num_pos = posAnnos_.size();
			Integer [] posAnnos = new Integer[num_pos];
			posAnnos = posAnnos_.toArray(posAnnos);
			// Randomly sample positive annotation
			Integer iid = posAnnos[rand.nextInt(num_pos)];
			// Sample negative annotation
			Integer jid =  this.negativeSampling(pid, iid);

			double hat_r_pi = VectorCalc.vectorInnerProduct(P[pid], Q[iid]);
			double hat_r_pj = VectorCalc.vectorInnerProduct(P[pid], Q[jid]);

			loss += -Math.log(1.0 / (1.0 + Math.exp(-(hat_r_pi - hat_r_pj)))); // -ln(sigmoid(x)), x = hat_r_ui-hat_r_uj
			double gradient = 1.0 / (1.0 + Math.exp(hat_r_pi - hat_r_pj));     // sigmoid(-x), x = hat_r_ui-hat_r_uj
				
			for(int k = 0; k < Config.K; ++k){
				P[pid][k] = P[pid][k] + Config.eta * (gradient * (Q[iid][k] - Q[jid][k]) - Config.lambdau * P[pid][k]);
				Q[iid][k] = Q[iid][k] + Config.eta * (gradient * P[pid][k] - Config.lambdai * Q[iid][k]);
				Q[jid][k] = Q[jid][k] + Config.eta * (- gradient * P[pid][k] - Config.lambdai * Q[jid][k]);
					
				loss += Config.lambdau*P[pid][k]*P[pid][k] + Config.lambdai*Q[iid][k]*Q[iid][k] +  Config.lambdai*Q[jid][k]*Q[jid][k];
			}
		}
		return loss;
	}
	
	Integer negativeSampling(Integer pid, Integer iid){
		Integer neg_img = 0;
		double pos_score = VectorCalc.vectorInnerProduct(P[pid], Q[iid]);
		// Sample score of negative annotation is larger than positive annotation
		for ( Integer aid = 1; aid < Config.annoNum; ++aid ){
		    if (this.trainMatrix.get(pid).contains(aid)) continue;
		    if (VectorCalc.vectorInnerProduct(P[pid], Q[aid]) + 1.0 > pos_score){
		        neg_img = aid;
		        break;
		    }
		}
		// No violate annotation
		if (neg_img == 0){
			Random rand = new Random(Config.randomSeed);
			while(trainMatrix.get(pid).contains(neg_img) || (this.testMatrix.get(pid)!=null&&this.testMatrix.get(pid).contains(neg_img)) || neg_img == 0) neg_img = rand.nextInt(Config.annoNum);
		}
		return neg_img;
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
			if (!this.trainMatrix.containsKey(pid)){removeList.add(pid);continue;}
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
				testImageAnnoPairs.add(new ImageAnnoPairUnit(pid, iid, jid)); // negative annotations cannot be in positive annotation set
			}
		}
		reader.close();
	}

	void initialize() {
		Random rand = new Random(10);
		
		P = new double[Config.imageNum][Config.K];
		for(int p = 0; p < Config.imageNum; ++p){
			P[p] = new double[Config.K];
			for(int k = 0; k < Config.K; ++k)
				P[p][k] = (rand.nextDouble() - 0.5) * 2.0 * Config.initDelta; // [-initDelta, initDelta]
		}
		
		Q = new double[Config.annoNum][Config.K];
		for(int i = 0; i < Config.annoNum; ++i){
			Q[i] = new double[Config.K];
			for(int k = 0; k < Config.K; ++k)
				Q[i][k] = (rand.nextDouble() - 0.5) * 2.0 * Config.initDelta; // [-initDelta, initDelta]
		}
	}

	public static void main(String[] args) throws IOException {
		Config.roundNum = 100;
		Config.eta = 0.1;
		Config.K = 100;
		Config.warp.driver();
	}
}
