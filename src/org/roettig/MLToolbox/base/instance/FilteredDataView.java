package org.roettig.MLToolbox.base.instance;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

public class FilteredDataView<T extends Instance> implements InstanceContainer<T>
{
	
	protected LabelSupplier lab_suppl;
	
	protected List<InstanceFilter<T>> filters = new ArrayList<InstanceFilter<T>>();
	
	protected List<T> data = new ArrayList<T>(); 
	protected List<T> unlab_data = new ArrayList<T>();
	
	public FilteredDataView()
	{
		
	}
	
	public void addAll(InstanceContainer<T> data)
	{
		for(T t: data)
		{
			if(accept(t))
				this.data.add(t);
		}
	}
	
	public void addFilter(InstanceFilter<T> filter)
	{
		filters.add(filter);
	}
	
	private boolean accept(T t)
	{
		for(InstanceFilter<T> filter: filters)
		{
			if(!filter.accept(t))
				return false;
		}
		return true;
	}

	@Override
	public Iterator<T> iterator()
	{
		return data.iterator();
	}

	@Override
	public void add(T t)
	{
		if(accept(t))
			this.data.add(t);
	}

	@Override
	public T get(int idx)
	{
		return data.get(idx);
	}

	@Override
	public int size()
	{
		return data.size();
	}

	@Override
	public LabelSupplier getLabelSupplier()
	{
		return lab_suppl;
	}

	@Override
	public void setLabelSupplier(LabelSupplier lab_suppl)
	{
		this.lab_suppl = lab_suppl;
	}

	@Override
	public void shuffle(Random rng)
	{
		Collections.shuffle(data,rng);
	}

	@Override
	public void dump()
	{
		for(Instance i: this.data)
		{
			System.out.println(i.getId());
		}
	}
	
	@Override
	public void setUnlabeledData(Iterable<T> data)
	{
		for(T t: data)
			unlab_data.add(t);
	}

	@Override
	public Collection<T> getUnlabeledData()
	{
		return unlab_data;
	}

	@Override
	public void clear()
	{
		// NOP
	}
}
