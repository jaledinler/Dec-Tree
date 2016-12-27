import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Instances;

public class DecisionTree {
    static int numOfAttributes;
	public static void main(String[] args) throws IOException {
		if(args.length != 3) {
			System.out.println(args.length);
			System.err.println("Usage: Java DecisionTree"
					+ "<train-set-file> <test-set-file> threshold");
			System.exit(1);;
		}
		BufferedReader train = new BufferedReader(
				new FileReader(args[0]));
		BufferedReader test = new BufferedReader(
				new FileReader(args[1]));
		Instances trainSet = new Instances(train);
		trainSet.setClassIndex(trainSet.numAttributes() - 1);
		DecisionTreeImpl decisionTree = new DecisionTreeImpl(trainSet,Integer.parseInt(args[2]));
		decisionTree.print(trainSet);
		Instances testSet = new Instances(test);
		decisionTree.printClassification(testSet);		
		testSet.setClassIndex(testSet.numAttributes() - 1);
		train.close();
		test.close();

	}

}
