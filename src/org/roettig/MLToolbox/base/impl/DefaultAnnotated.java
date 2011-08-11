package org.roettig.MLToolbox.base.impl;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.roettig.MLToolbox.base.Annotated;

/**
 * Default implementation of the Annotated interface.
 * 
 * @author roettig
 *
 */
public class DefaultAnnotated implements Annotated
{

	private Map<String,Object> data = new HashMap<String,Object>();
	
	public DefaultAnnotated()
	{	
	}
	
	@Override
	public void addProperty(String name, Object obj)
	{
		data.put(name, obj);
	}

	@Override
	public Object getProperty(String name)
	{
		return data.get(name);
	}

	@Override
	public void removeProperty(String name)
	{
		data.remove(name);
	}

	@Override
	public boolean hasProperty(String name)
	{
		return data.containsKey(name);
	}

	@Override
	public List<String> getPropertyNames()
	{
		List<String> names = new ArrayList<String>();
		for(String key: data.keySet())
		{
			names.add(key);
		}
		return names;
	}
	
}