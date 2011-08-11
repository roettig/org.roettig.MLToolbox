package org.roettig.MLToolbox.kernels;

import java.io.Serializable;

import org.roettig.MLToolbox.base.impl.DefaultParametrizedComposite;
import org.roettig.MLToolbox.base.instance.Instance;

/**
 * The KernelFunction class is the abstract base class for
 * all kernel functions defined on instances within MLToolbox.
 * @author roettig
 *
 * @param <T> type parameter of instance
 */
public abstract class KernelFunction<T extends Instance> extends DefaultParametrizedComposite implements Serializable
{
	private static final long serialVersionUID = -844323374869269027L;
	
	/**
	 * 
	 * gives the normalized kernel function value between objects x and y
	 * 
	 * @param x
	 * @param y
	 * @return double
	 * @throws Exception
	 */
	public final double computeN(T x, T y) throws Exception
	{
		return  compute(x,y)/(Math.sqrt(compute(x,x)*compute(y,y)));
	}

	/**
	 * gives the normalized kernel function value between objects x and y.
	 * This is the only method that has to be overridden in sub classes.
	 * 
	 * @param x
	 * @param y
	 * @return double 
	 * @throws Exception
	 */
	public abstract double compute(T x, T y) throws Exception;
	
	private boolean normalize = true;
	
	/**
	 * activate computation of normalized kernel function values.
	 * 
	 * @param flag
	 */
	public void doNormalize(boolean flag) 
	{
		normalize = flag;
	}
	
	/**
	 * are kernel values normalized.
	 * 
	 * @return flag
	 */
	public boolean isNormalized()
	{
		return normalize;
	}
}
