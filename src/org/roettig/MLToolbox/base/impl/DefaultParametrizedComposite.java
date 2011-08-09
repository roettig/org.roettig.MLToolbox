package org.roettig.MLToolbox.base.impl;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.roettig.MLToolbox.base.parameter.Parameter;
import org.roettig.MLToolbox.base.parameter.Parametrized;
import org.roettig.MLToolbox.base.parameter.ParametrizedComposite;

public class DefaultParametrizedComposite implements ParametrizedComposite, Serializable
{

	private List<Parametrized> ps = new ArrayList<Parametrized>();   
	
	@Override
	public List<Parameter<?>> getParameters()
	{
		List<Parameter<?>> ret = new ArrayList<Parameter<?>>();
		for(Parametrized p: ps)
		{	
			for(Parameter<?> param: p.getParameters())
				ret.add(param);
		}
		for(Parameter<?> param: params.values())
			ret.add(param);
		return ret;
	}
	
	public Parameter<?> getParameter(String name)
	{
		return params.get(name);
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
		return ps;
	}
	
	Map<String,Parameter<?>> params = new HashMap<String,Parameter<?>>();
	
	@Override
	public void registerParameter(String name, Parameter<?> param)
	{
		params.put(name,param);
	}
	
}
