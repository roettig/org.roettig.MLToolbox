package org.roettig.MLToolbox.kernels;

import org.roettig.MLToolbox.base.instance.InstanceContainer;
import org.roettig.MLToolbox.base.instance.PrimalInstance;
import org.roettig.MLToolbox.base.parameter.Parameter;

/**
 * this class implements the RBFKernel with parameter <i>gamma</i>. 
 * @author roettig
 *
 */
public class RBFKernel extends KernelFunction<PrimalInstance>
{
	private static final long serialVersionUID = -3724412740817065414L;

	public static String GAMMA = RBFKernel.class.getCanonicalName()+"_gamma";
	
	public RBFKernel()
	{
		Parameter<Double> C = new Parameter<Double>(GAMMA,new Double[]{0.1});
		this.registerParameter(GAMMA, C);
	}
	
	/**
	 * sets the allowed values for the parameter gamma.
	 * 
	 * @param gammas
	 */
	public void setGamma(Double... gammas)
	{
		Parameter<Double> C = new Parameter<Double>(GAMMA,gammas);
		this.registerParameter(C.getName(), C);
	}
	
	/*
	@Override
	public RBFKernel clone()
	{
		RBFKernel ret = null;
		ret = (RBFKernel) super.clone();
		return ret;
	}
	*/

	@Override
	public double compute(PrimalInstance x, PrimalInstance y)
	{		
		double xf[] = x.getFeatures();
		double yf[] = y.getFeatures();
		
		if(xf.length!=yf.length)
			throw new RuntimeException("Unsupported objects used in kernel evaluation: dimensions do not match");
		
		double d = 0.0;
		int    N = xf.length;
    	for(int i=0;i<N;i++)
    	{
    		d+=(xf[i]-yf[i])*(xf[i]-yf[i]);
    	}
    	
    	Parameter<?> param = this.getParameter(GAMMA);
    	double gamma = (Double) param.getCurrentValue();
    	return Math.exp( -gamma*d );
	}
	
	/**
	 * estimate a sensible initial value for the parameter gamma. 
	 * 
	 * @param data
	 * @return initial estimate
	 */
	public static double estimateGamma(InstanceContainer<PrimalInstance> data)
	{
		double g=0.0;
		int N = data.size();
		int c = 0;
		for(int i=0;i<N;i++)
		{
			PrimalInstance t1 = data.get(i);
			for(int j=i+1;j<N;j++)
			{
				PrimalInstance t2 = data.get(j);
				g+= PrimalInstance.squaredDistance(t1, t2);
				c++;
			}
		}
		g= g/c;
		return 1/(2.0*g);
	}
}