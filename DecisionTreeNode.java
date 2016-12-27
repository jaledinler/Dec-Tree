import java.util.LinkedList;
import java.util.List;

import weka.core.Attribute;
import weka.core.Instances;

public class DecisionTreeNode {
	int att_index;
	int parentAttributeValue;
	List<DecisionTreeNode> children;
	boolean terminal;
	int label;
	List<Attribute> usedAttributes;
	int fCount;
	int tCount;
	double split;
	DecisionTreeNode node;
	DecisionTreeNode(int label, int att_index, int parentAttributeValue, boolean terminal) {
		this.label = label;
		this.att_index = att_index;
		this.parentAttributeValue = parentAttributeValue;
		this.terminal = terminal;
		if (terminal) {
			children = null;
		} else {
			children = new LinkedList<DecisionTreeNode>();
		}
		usedAttributes = new LinkedList<Attribute>();
	}

	/**
	 * Add child to the node.
	 * 
	 * For printing to be consistent, children should be added
	 * in order of the attribute values as specified in the
	 * dataset.
	 */
	public void addChild(DecisionTreeNode child) {
		if (children != null) {
			children.add(child);
		}
	}
}
	
	