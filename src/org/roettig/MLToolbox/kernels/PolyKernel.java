package org.roettig.MLToolbox.kernels;

import org.roettig.MLToolbox.base.instance.PrimalInstance;
import org.roettig.MLToolbox.base.parameter.Parameter;

public class PolyKernel extends KernelFunction<PrimalInstance>
{

	public static String DEGREE = PolyKernel.class.getCanonicalName()+"_degree";
	
	public PolyKernel()
	{
		Parameter<Integer> D = new Parameter<Integer>(DEGREE,new Integer[]{2});
		this.registerParameter(DEGREE, D);
	}
	
	public void setDegrees(Integer... degrees)
	{
		Parameter<Integer> D = new Parameter<Integer>(DEGREE,degrees);
		this.registerParameter(D.getName(), D);
	}
	
	@Override
	public double compute(PrimalInstance x, PrimalInstance y) throws Exception
	{
		double xf[] = x.getFeatures();
		double yf[] = y.getFeatures();
		
		if(xf.length!=yf.length)
			throw new Exception("Unsupported objects used in kernel evaluation: dimensions do not match");
		
		double d = 0.0;
		int    N = xf.length;
    	for(int i=0;i<N;i++)
    	{
    		d+=(xf[i]*yf[i]);
    	}
    	
    	Parameter<?> param = this.getParameter(DEGREE);
    	int deg = (Integer) param.getCurrentValue();
    	return Math.pow(d+1,deg);
	}

}
