package org.roettig.MLToolbox.base.instance;

import java.util.Random;


public interface InstanceContainer<T extends Instance> extends Iterable<T>
{
	public void add(T t);
	public T get(int idx);
	public int size();
	public LabelSupplier getLabelSupplier();
	void setLabelSupplier(LabelSupplier lab_suppl);
	void shuffle(Random rng);
	void dump();
}
