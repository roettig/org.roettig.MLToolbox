package org.roettig.MLToolbox.base.instance;

import org.roettig.MLToolbox.base.label.Label;

public final class PrimalInstance extends Instance
{

	public PrimalInstance(Label label, double[] features)
	{
		super(label);
		this.features = new double[features.length]; 
		// defensive copying
		System.arraycopy(features, 0, this.features, 0, features.length);
	}
	
	public double[] getFeatures()
	{
		return features;
	}
		
	public int getNumberOfFeatures()
	{
		return features.length;
	}
	
	public static double squaredDistance(PrimalInstance x, PrimalInstance y)
	{
		double ret = 0.0;
		
		double[] x_fts = x.getFeatures();
		double[] y_fts = y.getFeatures();
		int N = x_fts.length;
		
		for(int i=0;i<N;i++)
		{
			double v = x_fts[i] - y_fts[i];
			ret += v*v;
		}
		return ret;
	}

	public static double distance(PrimalInstance x, PrimalInstance y)
	{
		double ret = 0.0;
		
		double[] x_fts = x.getFeatures();
		double[] y_fts = y.getFeatures();
		int N = x_fts.length;
		
		for(int i=0;i<N;i++)
		{
			double v = x_fts[i] - y_fts[i];
			ret += v*v;   
		}
		return Math.sqrt(ret);
	}
	
	private double[] features;

	@Override
	public PrimalInstance reassign(Label lab)
	{
		return new PrimalInstance(lab, this.features);
	}
}
