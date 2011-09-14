package org.roettig.MLToolbox.model.svm;

import java.util.HashMap;
import java.util.Map;

public class DefaultSVMTrainingParams implements SVMTrainingParams
{
	protected Map<String,Double> store = new HashMap<String,Double>();
	
	@Override
	public void setValue(String key, double value)
	{
		store.put(key,value);
	}

	@Override
	public double getValue(String key)
	{
		return store.get(key);
	}

	@Override
	public boolean hasKey(String key)
	{
		return store.containsKey(key);
	}

	@Override
	public void setFlag(String key)
	{
		store.put(key, 1.0);		
	}

	@Override
	public boolean getFlag(String key)
	{
		return store.containsKey(key);
	}

}
