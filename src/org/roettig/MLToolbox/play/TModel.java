package org.roettig.MLToolbox.play;

import java.util.List;

import org.roettig.MLToolbox.base.impl.DefaultParametrizedComposite;
import org.roettig.MLToolbox.base.parameter.Parameter;

public class TModel extends DefaultParametrizedComposite
{	
	private KernelFunction kfun;
	
	public TModel(KernelFunction kf)
	{
		kfun = kf;
		this.add(kfun);
		
	}
	
	public void setC(Double... Cs)
	{
		Parameter<Double> C = new Parameter<Double>(TModel.class.getCanonicalName()+"_C",Cs);
		this.registerParameter("C", C);
	}
	
	
	public static void main(String[] args)
	{
		
		KernelFunction kf = new KernelFunction();
		kf.setGamma(0.1,0.2,0.4,0.5);
		TModel          m  = new TModel(kf);
		m.setC(new Double[]{1.0,10.0,100.0});
		
		
		List<Parameter<?>> params = m.getParameters();
		for(Parameter<?> p: params)
		{
			System.out.println(p);
		}
	}

}
