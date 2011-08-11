package org.roettig.MLToolbox.base.label;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * FactorLabel is used for classification tasks.
 * 
 * @author roettig
 *
 */
public class FactorLabel extends Label
{
	
	public static Map<String,Double> classnames_to_double = new HashMap<String,Double>();
	public static Map<Double,Label>  double_to_label      = new HashMap<Double,Label>();
	
	private final String classname;
	private boolean freelabel = false;
	private double  val;
	
	/**
	 * returns for a given numerical value the corresponding label.
	 * 
	 * @param d numerical value
	 * 
	 * @return label
	 *
	 */
	public static Label fromDoubleValue(double d)
	{
		return double_to_label.get(d);
	}
	
	/**
	 * returns the numerical value of the label
	 */
	public double getDoubleValue()
	{
		if(freelabel)
			return val;
		else
			return classnames_to_double.get(getClassname());
	}
	
	/**
	 * factory method to create an unlabeled label.
	 * 
	 * @return label
	 */
	public static FactorLabel makeUnlabeled()
	{
		return new FactorLabel(null);
	}
	
	/**
	 * returns whether this label represents a unlabeled one.
	 * 
	 * @return unlabeled flag
	 */
	public boolean isUnlabeled()
	{
		return this.classname==null;
	}
		
	/**
	 * construct label from classname and numerical value.
	 * 
	 * <br/>
	 * 
	 * Note: This will become a free label, i.e. will not be administered statically by Label. 
	 * 
	 * @param classname
	 */
	public FactorLabel(String classname, double val)
	{
		freelabel = true;
		this.classname = classname;
		this.val = val;
	}
	
	/**
	 * construct label from classname.
	 * 
	 * <br/>
	 * 
	 * Note: This label will be administered statically by Label class.
	 * 
	 * @param classname
	 */
	public FactorLabel(String classname)
	{
		this.classname = classname;
		if(!classnames_to_double.containsKey(classname))
		{
			if(classnames_to_double.size()==0)
			{
				classnames_to_double.put(classname, 1.0);
				double_to_label.put(1.0,this);
			}
			else
			{
				double max_val = Collections.max(classnames_to_double.values())+1;
				classnames_to_double.put(classname, max_val);
				double_to_label.put(max_val,this);
			}
		}
	}
	
	/**
	 * return String represenation of this label.
	 */
	public String toString()
	{
		return String.format("%s",classname);
	}
	
	@Override
	public int hashCode()
	{
		return classname.hashCode();
	}

	@Override
	public boolean equals(Object obj)
	{
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		FactorLabel other = (FactorLabel) obj;
		if (classname == null)
		{
			if (other.classname != null)
				return false;
		} else if (!classname.equals(other.classname))
			return false;
		return true;
	}
	
	/**
	 * returns the classname of this label.
	 * 
	 * @return classname
	 */
	public String getClassname()
	{
		return classname;
	}
}
