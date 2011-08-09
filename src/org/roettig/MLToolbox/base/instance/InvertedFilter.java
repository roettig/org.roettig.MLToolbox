package org.roettig.MLToolbox.base.instance;

public class InvertedFilter<T extends Instance> implements InstanceFilter<T>
{
	
	private InstanceFilter<T> filter;
	
	public InvertedFilter(InstanceFilter<T> filter)
	{
		this.filter = filter;
	}
	
	@Override
	public boolean accept(T t)
	{
		return !filter.accept(t);
	}
}
