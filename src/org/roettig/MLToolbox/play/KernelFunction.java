package org.roettig.MLToolbox.play;

import java.util.List;

import org.roettig.MLToolbox.base.impl.DefaultParametrizedComposite;
import org.roettig.MLToolbox.base.parameter.Parameter;

public class KernelFunction extends DefaultParametrizedComposite
{

	//private DefaultParametrizedComposite ps = new DefaultParametrizedComposite(); 
	
	public KernelFunction()
	{
		
	}
	
	public void setGamma(Double... gammas)
	{
		Parameter<Double> gamma = new Parameter<Double>(KernelFunction.class.getCanonicalName()+"_gamma",gammas);
		this.registerParameter("gamma", gamma);
	}
	
	public static void main(String[] args)
	{
		KernelFunction kf = new KernelFunction();
		kf.setGamma(0.1,0.2,0.4,0.5);
		for(Parameter<?> p: kf.getParameters())
			System.out.println(p);
	}
	
	/*
	@Override
	public List<Parameter<?>> getParameters()
	{
		return ps.getParameters();
	}
	
	@Override
	public void add(Parametrized comp)
	{
		ps.add(comp);
	}

	@Override
	public void remove(Parametrized comp)
	{
		ps.remove(comp);
	}

	@Override
	public List<Parametrized> getChildren()
	{
		return ps.getChildren();
	}
	*/
}
