package org.roettig.MLToolbox.kernels;

import org.roettig.MLToolbox.base.instance.PrimalInstance;

/**
 * The LinearKernel computes the dot product between two primal instances.
 * 
 * @author roettig
 *
 */
public class LinearKernel extends KernelFunction<PrimalInstance>
{
	private static final long	serialVersionUID	= -8210757784279622459L;

	@Override
	public double compute(PrimalInstance x, PrimalInstance y) throws Exception
	{
		double[] xf = x.getFeatures();
		double[] yf = y.getFeatures();
		
		if(xf.length!=yf.length)
			throw new Exception("Unsupported objects used in kernel evaluation: dimensions do not match");
		
		double sum = 0.0;
		int N  = xf.length;
		for(int i=0;i<N;i++)
		{
			sum+=xf[i]*yf[i];
		}
		return sum;
	}

}
