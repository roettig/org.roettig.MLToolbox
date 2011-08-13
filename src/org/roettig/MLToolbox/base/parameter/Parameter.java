package org.roettig.MLToolbox.base.parameter;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * The Parameter class represents adjustable parameters within the MLToolbox.
 *  
 * <pre>
 * {@code
 * // parameter C1 will hold one value of 1.0
 * Parameter<Double> C1 = new Parameter<Double>("C1",new Double[]{1.0});
 * 
 * // parameter C2 will hold several allowed values 1.0,10.0,100.0
 * Parameter<Double> C2 = new Parameter<Double>("C2",new Double[]{1.0,10.0,100.0});
 * 
 * }
 * </pre>
 * 
 * @param <T> type of the encapsulated parameter (i.e. Double, ...)
 */
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

	/**
	 * get k-th allowed value.
	 * 
	 * @param k
	 * 
	 * @return value
	 */
	public T getValue(int k)
	{
		return values.get(k);
	}

	/**
	 * get current set value.
	 * 
	 * @return current value
	 */
	public T getCurrentValue()
	{
		return values.get(currValueIdx);
	}

	/**
	 * set current value to next allowed value.
	 * 
	 * @return next value
	 */
	public T getNextValue()
	{
		currValueIdx++;
		if(currValueIdx==values.size())
			currValueIdx=0;
		return getCurrentValue();
	}

	/**
	 * get number of allowed values.
	 * 
	 * @return number fo allowed values
	 */
	public int getSize()
	{
		return values.size();
	}

	/**
	 * set current value to i-th allowed value.
	 *  
	 * @param i 
	 */
	public void setCurrentValue(int i)
	{
		currValueIdx = i;
	}

	/**
	 * returns the name of the parameter.
	 * 
	 * @return name
	 */
	public String getName()
	{
		return name;
	}
	
	/**
	 * returns String representation of parameter.
	 */
	public String toString()
	{
		String ret = String.format("parameter [%s] values: ", this.getName());
		for(T val: values)
			ret += val+",";
		return ret.substring(0,ret.length()-1);
	}
}