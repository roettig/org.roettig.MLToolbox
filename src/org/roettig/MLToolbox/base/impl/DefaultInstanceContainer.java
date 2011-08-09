package org.roettig.MLToolbox.base.impl;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import org.roettig.MLToolbox.base.instance.Instance;
import org.roettig.MLToolbox.base.instance.InstanceContainer;
import org.roettig.MLToolbox.base.instance.LabelSupplier;
import org.roettig.MLToolbox.base.instance.PrimalInstance;
import org.roettig.MLToolbox.base.label.FactorLabel;
import org.roettig.MLToolbox.base.label.Label;

public class DefaultInstanceContainer<T extends Instance> implements InstanceContainer<T>
{
	private List<T> data;
	private LabelSupplier lab_suppl;
	
	public DefaultInstanceContainer()
	{
		this.data = new ArrayList<T>();
	}
	
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
	
	public void add(T t)
	{
		data.add(t);
	}
	
	public void clear()
	{
		this.data.clear();
	}
	
	@Override
	public Iterator<T> iterator()
	{
		return this.data.iterator();
	}
	
	public void shuffle(Random rng)
	{
		Collections.shuffle(data, rng);
	}
	
	public static void main(String[] args)
	{
		Label lab1 = new FactorLabel("class1");

		
		double[] fts1 = new double[]{1.0,2.0};		
		
		PrimalInstance pi1 = new PrimalInstance(lab1,fts1);
		PrimalInstance pi2 = new PrimalInstance(lab1,fts1);
		PrimalInstance pi3 = new PrimalInstance(lab1,fts1);
		PrimalInstance pi4 = new PrimalInstance(lab1,fts1);
		PrimalInstance pi5 = new PrimalInstance(lab1,fts1);
		
		List<PrimalInstance> data = new ArrayList<PrimalInstance>();
		data.add(pi1);
		data.add(pi2);
		data.add(pi3);
		data.add(pi4);
		data.add(pi5);
		

		
		DefaultInstanceContainer<PrimalInstance> iv = new DefaultInstanceContainer<PrimalInstance>(data);
		
		for(PrimalInstance pi: iv)
		{
			System.out.println(pi.getId());
		}
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
	
	public void dump()
	{
		for(Instance i: this.data)
		{
			System.out.println(i.getId());
		}
	}
}
