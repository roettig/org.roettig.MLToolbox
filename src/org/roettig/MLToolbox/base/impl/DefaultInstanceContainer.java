package org.roettig.MLToolbox.base.impl;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.roettig.MLToolbox.base.instance.Instance;
import org.roettig.MLToolbox.base.instance.InstanceContainer;
import org.roettig.MLToolbox.base.instance.LabelSupplier;
import org.roettig.MLToolbox.base.instance.PrimalInstance;
import org.roettig.MLToolbox.base.label.FactorLabel;
import org.roettig.MLToolbox.base.label.Label;

/**
 * Default implementation of the InstanceContainer interface.
 * 
 * @author roettig
 *
 * @param <T> type parameter of instance
 * 
 */
public class DefaultInstanceContainer<T extends Instance> implements InstanceContainer<T>
{
	private List<T> data = new ArrayList<T>();
	private List<T> unlab_data = new ArrayList<T>();
	private LabelSupplier lab_suppl;
	
	/**
	 * creates an empty instance container.
	 */
	public DefaultInstanceContainer()
	{
		this.data = new ArrayList<T>();
	}
	
	/**
	 * creates an instance container holding the supplied instances.
	 * 
	 * @param data
	 */
	public DefaultInstanceContainer(Collection<T> data)
	{
		this();
		Iterator<T> iter = data.iterator();
		while(iter.hasNext())
		{
			T t = iter.next();
			this.data.add(t);
		}		
	}
	
	@Override
	public void add(T t)
	{
		data.add(t);
	}
	
	@Override
	public void clear()
	{
		this.data.clear();
	}
	
	@Override
	public Iterator<T> iterator()
	{
		return this.data.iterator();
	}
	
	@Override
	public void shuffle(Random rng)
	{
		Collections.shuffle(data, rng);
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
	
	public void setLabelSupplier(LabelSupplier lab_suppl)
	{
		this.lab_suppl = lab_suppl;
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
	public boolean isFactorLabelled()
	{
		if(this.data.size()>0)
		{
			Label lab = this.data.get(0).getLabel();
			return (lab instanceof FactorLabel);
		}
		return false;
	}
	
	public static <T extends Instance> int getNumberOfFactors(InstanceContainer<T> data)
	{
		Set<Label> factors = new HashSet<Label>();
		for(Instance i: data)
		{
			Label lab = i.getLabel();
			if(lab instanceof FactorLabel)
			{
				factors.add(lab);
			}
		}
		return factors.size();
	}
}
