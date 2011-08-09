package org.roettig.MLToolbox.base.label;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class LabelFactory
{
	public static Map<String,Double> classnames_to_double = new HashMap<String,Double>();
	public static Map<String,Label>  factor_labels        = new HashMap<String,Label>();
	
	public static Label getFactorLabel(String classname)
	{
		if(factor_labels.containsKey(classname))
			return factor_labels.get(classname);
		
		FactorLabel lab = new FactorLabel(classname);
		factor_labels.put(classname, lab);
		
		if(!classnames_to_double.containsKey(classname))
		{
			if(classnames_to_double.size()==0)
			{
				classnames_to_double.put(classname, 1.0);
			}
			else
			{
				double max_val = Collections.max(classnames_to_double.values())+1;
				classnames_to_double.put(classname, max_val);
				
			}
		}
		return lab; 
	}
	
	//public static double getLabelValue(Label lab)
}
