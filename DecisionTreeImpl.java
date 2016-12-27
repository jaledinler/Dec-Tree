import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class DecisionTreeImpl {
	DecisionTreeNode root;
	List<Double> candidateSplits;
	Map<String, Integer> labelCount;
	Map<String,Double> splitValueByAttribute;
	int maxAttributeIndex;
	int labelIndex;
	Attribute label;
	DecisionTreeImpl(Instances instances, int m) {
		splitValueByAttribute = new TreeMap<String,Double>();
		instances.setClassIndex(instances.numAttributes()-1);
		label = instances.attribute(instances.numAttributes()-1);
		labelIndex = instances.numAttributes()-1;
		maxAttributeIndex = -1;
		root = new DecisionTreeNode(-1,-1,-1,false);
		build(instances,root,null,m);
	}
	public int MajorityVote(Instances instances) {
		int Tcount = 0;
		int Fcount = 0;
		Iterator<Instance> itr = instances.iterator();
		while(itr.hasNext())
		{
			Instance nextInstance = itr.next();
			int index = (int) nextInstance.value(labelIndex);
			if(index == 0)
				Fcount++;
			else
				Tcount++;			

		}
		if(Tcount >= Fcount)
			return 1;
		else
			return 0;
	}

	public void build(Instances instances, DecisionTreeNode node, DecisionTreeNode parent, int m) {
		if(instances == null || node == null)
			throw new IllegalArgumentException("Build");
		int f,t;
		if(node.label != -1) {
			node.terminal = true;
			return;
		}
		if(node.label == -1)
			node.terminal = false;
		if(node.terminal == true && node.node == root && root.fCount == root.tCount) {
			node.label = 0;
			return;
		}
		if(instances.numInstances() < m || instances.numAttributes() == 0) {
			if(node.fCount == node.tCount) {
				f = (node.node).fCount;
				t = (node.node).tCount;
				if(f > t)
					node.label = 0;
				if(t > f)
					node.label = 1;
			}
			else {
				node.fCount = countLabels(instances).get(instances.attribute(labelIndex).value(0));
				node.tCount = countLabels(instances).get(instances.attribute(labelIndex).value(1));
				node.label = MajorityVote(instances);
			}
			node.terminal = true;
			return;
		}
		int fCount = 0;
		int tCount = 0;
		for(int i = 0; i < instances.numInstances(); i++)
		{
			if((int)instances.get(i).value(labelIndex) == 0) 
				fCount++;
			else
				tCount++;
		}
		if(fCount == instances.numInstances()) {
			node.fCount = fCount;
			node.tCount = 0;
			node.label = 0;
			node.terminal = true;
			return;
		}
		if(tCount == instances.numInstances()) {
			node.fCount = 0;
			node.tCount = tCount;
			node.label = 1;
			node.terminal = true;
			return;
		}
		double infoGain;
		double maxInfoGain = 0;
		for(int i = 0; i < instances.numAttributes()-1; i++)
		{
			Attribute attribute = instances.attribute(i);
			if(!node.usedAttributes.contains(attribute)) {
				if(attribute.isNominal()) {
					infoGain = infoGainNom(instances,attribute);
					if(infoGain > maxInfoGain) {
						maxInfoGain = infoGain;
						maxAttributeIndex = i;
					}
				}
				if(attribute.isNumeric()) {
					infoGain = infoGainNum(instances,attribute);
					if(infoGain > maxInfoGain) {
						maxInfoGain = infoGain;
						maxAttributeIndex = i;
					}
				}
			}
		}
		if(maxInfoGain == 0) {
			node.terminal = true;
			node.label = MajorityVote(instances);
			return;
		}
		Attribute attribute = instances.attribute(maxAttributeIndex);
		node.fCount = countLabels(instances).get(instances.attribute(labelIndex).value(0));
		node.tCount = countLabels(instances).get(instances.attribute(labelIndex).value(1));
		node.att_index = maxAttributeIndex;
		if(!node.usedAttributes.contains(attribute) && attribute.isNominal())
			node.usedAttributes.add(attribute);
		if(node.fCount != node.tCount)
			node.node = node;
		node.children = new LinkedList<DecisionTreeNode>();
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		for(int j = 0; j < instances.numAttributes();j++) {
			attributes.add(instances.attribute(j));
		}
		DecisionTreeNode child;
		if(attribute.isNominal()) {
			for(int i = 0; i < attribute.numValues(); i++) {
				child = new DecisionTreeNode(-1,-1,i,false);
				child.usedAttributes.add(attribute);
				if(node.fCount != node.tCount)
					child.node = node;
				List<Instance> childInstanceList = new LinkedList<Instance>();
				Iterator<Instance> itr = instances.iterator();
				while(itr.hasNext()) {
					Instance instance = itr.next();					
					int index = (int) instance.value(node.att_index);
					if(index == i)
					{
						childInstanceList.add(instance);
					}
				}
				int size = childInstanceList.size();
				if(size == 0) {
					if(node.node.fCount > node.node.tCount)
						child.label = 0;
					else
						child.label = 1;
					child.terminal = true;
				}
				Instances childInstances  = new Instances("c",attributes,size);
				Iterator<Instance> itr1 = childInstanceList.iterator();
				while(itr1.hasNext()) {
					Instance instance = itr1.next();
					childInstances.add(instance);
				}
				if(!childInstances.isEmpty()) {
					child.fCount = countLabels(childInstances).get(childInstances.attribute(labelIndex).value(0));
					child.tCount = countLabels(childInstances).get(childInstances.attribute(labelIndex).value(1));
				}
				if(child.fCount == child.tCount && child.node == null) {
					child.node = node.node;
				}
				if(child.fCount != child.tCount) {
					child.node = child;
				}

				if(child.label != -1)
					child.terminal = true;
				node.children.add(child);
				if(parent == null)
				{
					parent = node;
					build(childInstances,child,root,m);
				}
				else {
					if(parent.children.isEmpty())
						parent.children.add(node);
					else {
						for(int j = 0; j < parent.children.size(); j++) {
							DecisionTreeNode nextChild = parent.children.get(j);
							if(nextChild.att_index == node.att_index && nextChild.parentAttributeValue == node.parentAttributeValue) {
								int index = parent.children.indexOf(nextChild);
								parent.children.remove(index);
								parent.children.add(index,node);
							}
						}						
					}
					build(childInstances,child,node,m);
				}
			}
		}
		DecisionTreeNode child1,child2;
		if(attribute.isNumeric()) {
			List<Instance> childInstanceList1 = new LinkedList<Instance>();
			child1 = new DecisionTreeNode(-1,-1,0,false);
			child1.usedAttributes = node.usedAttributes;
			if(node.fCount != node.tCount)
				child1.node =node;
			child1.split = splitValueByAttribute.get(attribute.name());
			Iterator<Instance> itr1 = instances.iterator();

			while(itr1.hasNext()) {
				Instance instance = itr1.next();
				if(instance.value(maxAttributeIndex) <= splitValueByAttribute.get(attribute.name()))
					childInstanceList1.add(instance);
			}				

			int size1 = childInstanceList1.size();
			Instances childInstances1  = new Instances("c",attributes,size1);
			Iterator<Instance> itr3 = childInstanceList1.iterator();
			if(size1 == 0) {
				if(node.node.fCount > node.node.tCount)
					child1.label = 0;
				else
					child1.label = 1;
				child1.terminal = true;
			}
			while(itr3.hasNext()) {
				Instance instance = itr3.next();
				childInstances1.add(instance);
			}
			if(!childInstances1.isEmpty()) {
				child1.fCount = countLabels(childInstances1).get(childInstances1.attribute(labelIndex).value(0));
				child1.tCount = countLabels(childInstances1).get(childInstances1.attribute(labelIndex).value(1));
			}
			if(child1.fCount == child1.tCount && child1.node == null) {
				child1.node = node.node;
			}
			if(child1.fCount != child1.tCount) {
				child1.node = child1;
			}

			List<Instance> childInstanceList2 = new LinkedList<Instance>();
			child2 = new DecisionTreeNode(-1,-1,1,false);
			child2.usedAttributes = node.usedAttributes;
			if(node.fCount != node.tCount)
				child2.node = node;
			child2.split = splitValueByAttribute.get(attribute.name());
			Iterator<Instance> itr2 = instances.iterator();		

			while(itr2.hasNext()) {
				Instance instance = itr2.next();
				if(instance.value(maxAttributeIndex) > splitValueByAttribute.get(attribute.name()))
					childInstanceList2.add(instance);
			}

			int size2 = childInstanceList2.size();
			if(size2 == 0) {
				if(node.node.fCount > node.node.tCount)
					child2.label = 0;
				else
					child2.label = 1;
				child2.terminal = true;
			}
			Instances childInstances2  = new Instances("c",attributes,size2);
			Iterator<Instance> itr4 = childInstanceList2.iterator();
			while(itr4.hasNext()) {
				Instance instance = itr4.next();
				childInstances2.add(instance);
			}
			if(!childInstances2.isEmpty()) {
				child2.fCount = countLabels(childInstances2).get(childInstances2.attribute(labelIndex).value(0));
				child2.tCount = countLabels(childInstances2).get(childInstances2.attribute(labelIndex).value(1));
			}
			if(child2.fCount == child2.tCount && child2.node == null) {
				child2.node = node.node;
			}
			if(child2.fCount != child2.tCount) {
				child2.node = child2;
			}

			if(child1.label != -1)
				child1.terminal = true;

			if(child2.label != -1)
				child2.terminal = true;

			node.children.add(child1);
			node.children.add(child2);


			if(parent == null)
			{
				parent = node;
				build(childInstances1,child1,root,m);
				build(childInstances2,child2,root,m);
			}
			else {
				if(parent.children.isEmpty())
					parent.children.add(node);
				else {
					for(int i = 0; i < parent.children.size();i++) {
						DecisionTreeNode nextChild = parent.children.get(i);
						if(nextChild.att_index == node.att_index && nextChild.parentAttributeValue == node.parentAttributeValue) {
							int index = parent.children.indexOf(nextChild);
							parent.children.remove(index);
							parent.children.add(index,node);
						}
					}
				}
				build(childInstances1,child1,node,m);
				build(childInstances2,child2,node,m);
			}
		}
	}
	public void printClassification(Instances instances) {
		if(instances == null)
			throw new IllegalArgumentException("");
		List<Integer> classification = classify(instances);
		int count = 0;
		System.out.print("<Predictions for the Test Set Instances>"+"\r\n");
		for(int i = 0; i < instances.numInstances(); i++) {
			Instance instance = instances.get(i);
			System.out.print(String.valueOf(i+1)+": Actual: "+label.value((int)instance.value(labelIndex))+" Predicted: "+label.value(classification.get(i))+"\r\n");
			if(classification.get(i) == (int)instance.value(labelIndex))
				count++;
		}
		System.out.print("Number of correctly classified: "+String.valueOf(count)+" Total number of test instances: "+String.valueOf(instances.numInstances())+"\r\n");
	}
	public List<Integer> classify(Instances instances) {
		if(instances == null)
			throw new IllegalArgumentException("");
		int label;
		List<Integer> classification = new LinkedList<Integer>();
		for(int i = 0 ; i < instances.numInstances(); i++) {
			Instance instance = instances.get(i);
			label = classify(instance,root);
			classification.add(i,label);
		}
		return classification;
	}
	public int classify(Instance instance, DecisionTreeNode node) {
		if(instance == null || node == null)
			throw new IllegalArgumentException("");

		int label = 0;
		if(node.terminal) {
			label = node.label;
		}

		while(node.att_index != -1){
			Attribute attribute = instance.attribute(node.att_index);
			if(attribute.isNominal()) {
				for(int i = 0; i < node.children.size(); i++) {
					if((int)instance.value(attribute) == i) {
						node = node.children.get(i);
						return classify(instance,node);
					}
				}
			}

			if(attribute.isNumeric()) {

				if(instance.value(attribute) <= node.children.get(0).split) {
					return classify(instance,node.children.get(0));
				}
				else {
					return classify(instance,node.children.get(1));
				}
			}
		}
		return label;
	}
	void determineCandidateNumericSplits(Instances instances, final Attribute att) {
		if(!att.isNumeric()) {
			throw new IllegalArgumentException(att+ "is not numeric.");
		}
		candidateSplits = new LinkedList<Double>();
		List<List<Instance>> partitionList = new LinkedList<List<Instance>>();

		for(int i = 0; i < instances.numInstances(); i++) {
			int count = 0;
			double value = instances.get(i).value(att);
			Iterator<List<Instance>> partList_itr = partitionList.iterator();
			while(partList_itr.hasNext()) {
				List<Instance> partition = partList_itr.next();
				Instance instance = partition.get(0);
				if(instance.value(att) == value) {
					int index = partitionList.indexOf(partition);
					partition.add(instances.get(i));
					partitionList.remove(index);
					partitionList.add(index,partition);
					break;
				}
				else
				{
					count++;
				}
			}
			if(count == partitionList.size()){
				List<Instance> newPartition = new LinkedList<Instance>();
				newPartition.add(instances.get(i));
				partitionList.add(newPartition);
			}
		}

		Collections.sort(partitionList, new Comparator<List<Instance>>() {
			public int compare(List<Instance> list1, List<Instance> list2) {
				if(list1 == null || list2 == null) {
					throw new IllegalArgumentException("");
				}
				else if(list1.get(0).value(att) > list2.get(0).value(att))
					return 1;
				return -1;
			}
		}
				);
		int count1 = 0;
		for(int i = 0; i < partitionList.size()-1; i++) {
			List<Instance> partition1 = partitionList.get(i);
			Iterator<Instance> itr1 = partition1.iterator();
			while(itr1.hasNext()) {
				Instance instance1 = itr1.next();
				List<Instance> partition2 = partitionList.get(i+1);
				Iterator<Instance> itr2 = partition2.iterator();
				while(itr2.hasNext()) {
					Instance instance2 = itr2.next();
					double split = (double)instance1.value(att)/2+(double)instance2.value(att)/2;
					if(instance1.value(labelIndex) != instance2.value(labelIndex) &&
							!candidateSplits.contains(split)) {

						candidateSplits.add(split);
						count1++;
						break;
					}
				}
				if(count1 > 0)
					break;
			}
			count1=0;
		}
	}

	public double calculateEntropy(int count, int numOfInstances) {
		double p = (double)count/numOfInstances;
		if(p != 0)
			return -(p*Math.log10(p))/Math.log10(2);
		return 0;
	}
	public double infoGainNum(Instances instances, Attribute att) {
		if(instances == null || att == null)
			throw new IllegalArgumentException("calculateTotalEntropyNum");
		double totalEntropy = calculateTotalEntropy(instances);
		double max_InfoGain = 0;
		double entropy_l = 0;
		double entropy_m = 0;
		double splitValue=0;
		determineCandidateNumericSplits(instances,att);
		Map<String,Integer> labelCountsByValue = new TreeMap<String,Integer>();
		for(int i = 0; i < candidateSplits.size(); i++) {
			double split_value = candidateSplits.get(i);
			String str = String.valueOf(split_value);
			Iterator<Instance> itr = instances.iterator();
			while(itr.hasNext()) {
				Instance instance = itr.next();
				int labelValue = (int) instance.value(labelIndex);
				if(instance.value(att) <=  split_value ) {
					for(int j = 0; j < label.numValues(); j++) {						
						if(labelValue == j) {
							String str1 = "l"+","+str+","+String.valueOf(j);
							if(labelCountsByValue.containsKey(str1))
								labelCountsByValue.put(str1, labelCountsByValue.get(str1)+1);
							else
								labelCountsByValue.put(str1, 1);	
						}
					}
				}
				else {
					for(int j = 0; j < label.numValues(); j++) {						
						if(labelValue == j) {
							String str1 = "m"+","+str+","+String.valueOf(labelValue);
							if(labelCountsByValue.containsKey(str1))
								labelCountsByValue.put(str1, labelCountsByValue.get(str1)+1);
							else
								labelCountsByValue.put(str1, 1);	
						}
					}
				}
			}
		}
		int total_less = 0;
		int total_more = 0;
		Set<Map.Entry<String,Integer>> set = labelCountsByValue.entrySet();

		for(int i = 0; i < candidateSplits.size(); i++) {
			String str = String.valueOf(candidateSplits.get(i));
			Iterator<Map.Entry<String,Integer>> itr1 = set.iterator();
			while(itr1.hasNext()) {
				Map.Entry<String,Integer> entry = itr1.next();
				String[] strarr = entry.getKey().split(",");
				int labelCount = entry.getValue();
				if(strarr[0].equals("l") && strarr[1].equals(str)) 
					total_less+=labelCount;	
				if(strarr[0].equals("m") && strarr[1].equals(str)) 
					total_more+=labelCount;	
			}
			Iterator<Map.Entry<String,Integer>> itr2 = set.iterator();

			while(itr2.hasNext()) {
				Map.Entry<String,Integer> entry = itr2.next();
				String[] strarr = entry.getKey().split(",");
				int labelCount = entry.getValue();
				for(int j = 0; j < label.numValues(); j++)
				{
					if(strarr[0].equals("l") &&
							strarr[1].equals(str) && strarr[2].equals(String.valueOf(j)) ) {
						entropy_l += calculateEntropy(labelCount,total_less);

					}
					if(strarr[0].equals("m") &&
							strarr[1].equals(str) && strarr[2].equals(String.valueOf(j)) ) {
						entropy_m += calculateEntropy(labelCount,total_more);

					}
				}
			}
			entropy_l = ((double)total_less/instances.numInstances())*entropy_l;
			entropy_m = ((double)total_more/instances.numInstances())*entropy_m;
			double entropy = entropy_l+entropy_m;
			double infoGain = totalEntropy-entropy;
			if(infoGain > max_InfoGain) {
				max_InfoGain = infoGain;
				splitValue = candidateSplits.get(i);
			}
			if(infoGain == max_InfoGain && splitValue > candidateSplits.get(i)) {
				max_InfoGain = infoGain;
				splitValue = candidateSplits.get(i);
			}
			total_less = 0;
			total_more = 0;
			entropy_l = 0;
			entropy_m = 0;			
		}
		splitValueByAttribute.put(att.name(),splitValue);
		return max_InfoGain;
	}

	public double infoGainNom(Instances instances, Attribute att) {
		if(instances == null)
			throw new IllegalArgumentException("calculateTotalEntropy");
		double entropy = 0;
		double totalEntropy = calculateTotalEntropy(instances);
		Map<String,Integer> labelCountsByValue = new TreeMap<String,Integer>();
		Map<String,Double> entropyByValue = new TreeMap<String,Double>();
		for(int i = 0; i < att.numValues(); i++) {
			Iterator<Instance> itr = instances.iterator();
			while(itr.hasNext()) {
				Instance instance = itr.next();
				int labelValue = (int) instance.value(labelIndex);
				if((int)instance.value(att) == i ) {
					for(int j = 0; j < label.numValues(); j++) {						
						if(j== labelValue) {
							if(labelCountsByValue.containsKey(i+","+String.valueOf(j)))
								labelCountsByValue.put(String.valueOf(i)+","+String.valueOf(j), labelCountsByValue.get(i+","+String.valueOf(j))+1);
							else
								labelCountsByValue.put(String.valueOf(i)+","+String.valueOf(j), 1);	
						}
					}
				}
			}
		}
		int total = 0;
		Set<Map.Entry<String,Integer>> set = labelCountsByValue.entrySet();

		for(int i = 0; i < att.numValues(); i++) {
			Iterator<Map.Entry<String,Integer>> itr1 = set.iterator();
			while(itr1.hasNext()) {
				Map.Entry<String,Integer> entry = itr1.next();
				String[] strarr = entry.getKey().split(",");
				int labelCount = entry.getValue();
				if(strarr[0].equals(String.valueOf(i))) 
					total+=labelCount;				
			}
			Iterator<Map.Entry<String,Integer>> itr2 = set.iterator();
			while(itr2.hasNext()) {
				Map.Entry<String,Integer> entry = itr2.next();
				String[] strarr = entry.getKey().split(",");
				int labelCount = entry.getValue();
				for(int j = 0; j < label.numValues(); j++)
				{
					if(strarr[1].equals(String.valueOf(j)) &&
							strarr[0].equals(String.valueOf(i))) {
						entropy += calculateEntropy(labelCount,total);

					}
				}
			}
			entropy = ((double)total/instances.numInstances())*entropy;
			entropyByValue.put(String.valueOf(i), entropy);
			entropy = 0;
			total = 0;
		}
		entropy = 0;
		Set<Map.Entry<String,Double>> set1 = entropyByValue.entrySet();
		Iterator<Map.Entry<String,Double>> itr3 = set1.iterator();
		while(itr3.hasNext()) {
			Map.Entry<String,Double> entry = itr3.next();
			entropy+=entry.getValue();
		}
		double infoGain = totalEntropy-entropy;
		return infoGain;
	}
	public double calculateTotalEntropy(Instances instances) {
		if(instances == null)
			throw new IllegalArgumentException("calculateTotalEntropy");
		labelCount = countLabels(instances);
		Set<Map.Entry<String, Integer>> set = labelCount.entrySet();
		Iterator<Map.Entry<String,Integer>> set_itr = set.iterator();
		List<Integer> valueEntropy = new LinkedList<Integer>();
		int total = 0;
		double entropy = 0;		
		while(set_itr.hasNext()){
			Map.Entry<String,Integer> entry = set_itr.next();
			int labelcount = entry.getValue();				
			valueEntropy.add(labelcount);
			total+= labelcount;
		}

		for(int i = 0; i < valueEntropy.size(); i++)
		{
			entropy+=calculateEntropy(valueEntropy.get(i),total);
		}	
		return entropy;
	}

	public Map<String,Integer> countLabels(Instances instances) {
		if(instances == null)
			throw new IllegalArgumentException("countLabels");
		labelCount = new TreeMap<String,Integer>();
		for(int j = 0; j < label.numValues(); j++) 
			labelCount.put(label.value(j),0);

		for(int i = 0; i < label.numValues(); i++) {
			String value = label.value(i);
			Iterator<Instance> itr = instances.iterator();
			while(itr.hasNext()) {
				Instance instance = itr.next();
				int  temp = (int) instance.value(labelIndex);
				if(i == temp) {				
					//					if(labelCount.containsKey(value))
					labelCount.put(String.valueOf(value), labelCount.get(String.valueOf(value))+1);
					//					else
					//						labelCount.put(String.valueOf(value), 1);
				}
			}
		}
		return labelCount;
	}
	public void print(Instances instances)
	{
		printTreeNode(instances,root, null, 0);
	}

	/**
	 * Prints the subtree of the node
	 * with each line prefixed by 4 * k spaces.
	 */
	public void printTreeNode(Instances instances,DecisionTreeNode p, DecisionTreeNode parent, int k)
	{
		StringBuilder sb = new StringBuilder();
		for (int i = 1; i < k; i++)
		{
			sb.append("|\t");
		}

		if(parent !=null)
		{
			Attribute parentAttribute = instances.attribute(parent.att_index);
			if(parentAttribute.isNominal()){
				String value = instances.attribute(parent.att_index).name()+" = "+parentAttribute.value(p.parentAttributeValue)+" ["+String.valueOf(p.fCount)+" "+String.valueOf(p.tCount)+"]";
				sb.append(value);
			}
			if(parentAttribute.isNumeric()) {
				if(p.parentAttributeValue == 0) {
					String value = instances.attribute(parent.att_index).name()+parentAttribute.value(0)+" <= "+String.format("%.6f",p.split)+" ["+String.valueOf(p.fCount)+" "+String.valueOf(p.tCount)+"]";
					sb.append(value);
				}
				if(p.parentAttributeValue == 1) {
					String value = instances.attribute(parent.att_index).name()+parentAttribute.value(1)+" > "+String.format("%.6f",p.split)+" ["+String.valueOf(p.fCount)+" "+String.valueOf(p.tCount)+"]";
					sb.append(value);
				}
			}

		}			

		if (p.terminal)
		{
			sb.append(": "+instances.classAttribute().value(p.label));
			System.out.print(sb.toString()+"\r\n");
		} else
		{
			if(!p.equals(root))
				System.out.print(sb.toString()+"\r\n");
			for (DecisionTreeNode child : p.children)
			{
				printTreeNode(instances,child, p, k + 1);
			}
		}
	}
}




