package org.roettig.MLToolbox.base.parameter;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class Parameter<T> implements Cloneable, Serializable
{

	private static final long serialVersionUID = 7643340732060278892L;

	private List<T> values;
	private int currValueIdx=0;
	private String name;


	@SuppressWarnings("unchecked")
	public Parameter<T> clone()
	{
		Parameter<T> ret = null;
		try 
		{
			ret = (Parameter<T>) super.clone();
		} 
		catch (CloneNotSupportedException e) 
		{
			// this shouldn't happen, since we are Cloneable
			throw new InternalError();
		} 
		return ret; 
	}


	public Parameter(String _name, List<T> vals)
	{
		name   = _name;
		values = new ArrayList<T>();
		for(T v: vals)
		{
			values.add(v);
		}
	}

	public Parameter(String _name, T[] vals)
	{
		name   = _name;
		values = new ArrayList<T>();
		for(T v: vals)
		{
			values.add(v);
		}
	}

	public T getValue(int k)
	{
		return values.get(k);
	}

	public T getCurrentValue()
	{
		return values.get(currValueIdx);
	}

	public T getNextValue()
	{
		currValueIdx++;
		if(currValueIdx==values.size())
			currValueIdx=0;
		return getCurrentValue();
	}

	public int getSize()
	{
		return values.size();
	}

	public void setCurrentValue(int idx)
	{
		currValueIdx = idx;
	}

	public String getName()
	{
		return name;
	}
	
	public String toString()
	{
		String ret = String.format("parameter [%s] values: ", this.getName());
		for(T val: values)
			ret += val+",";
		return ret.substring(0,ret.length()-1);
			
	}
}