package org.roettig.MLToolbox.base.label;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;


public class FactorLabel extends Label
{
	
	public static Map<String,Double> classnames_to_double = new HashMap<String,Double>();
	public static Map<Double,Label>  double_to_label      = new HashMap<Double,Label>();
	
	private final String classname;
	
	public static Label fromDoubleValue(double d)
	{
		return double_to_label.get(d);
	}
	
	public double getDoubleValue()
	{
		return classnames_to_double.get(getClassname());
	}
	
	public static FactorLabel makeUnlabeled()
	{
		return new FactorLabel(null);
	}
	
	public boolean isUnlabeled()
	{
		return this.classname==null;
	}
		
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
	
	public String getClassname()
	{
		return classname;
	}
}
