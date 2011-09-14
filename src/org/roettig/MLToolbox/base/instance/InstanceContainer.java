package org.roettig.MLToolbox.base.instance;

import java.io.Serializable;
import java.util.Collection;
import java.util.Random;

/**
 * The InstanceContainer interfaces defines a minimal set of methods to be able to work
 * with collections of instances.
 * 
 * <br/>
 * 
 * Note: You can make use of the DefaultInstanceContainer.
 * 
 * @author roettig
 *
 * @param <T> generic type of instances stored 
 */
public interface InstanceContainer<T extends Instance> extends Iterable<T>, Serializable
{
	/**
	 * add an instance to the container.
	 * 
	 * @param t
	 */
	public void add(T t);
	
	/**
	 * get the i-th instance.
	 * 
	 * @param idx
	 * 
	 * @return instance
	 */
	public T get(int idx);
	
	/**
	 * get number of stored instance in this container.
	 * 
	 * @return size
	 */
	public int size();
	
	/**
	 * remove all instances from this container.
	 */
	public void clear();
	
	public LabelSupplier getLabelSupplier();
	void setLabelSupplier(LabelSupplier lab_suppl);
	
	/**
	 * randomly shuffle the list of instances in this container.
	 * 
	 * @param rng a source of randomness
	 */
	void shuffle(Random rng);
	
	/**
	 * dump information about this container. 
	 */
	void dump();
	
	/**
	 * add unlabeled data to this container.
	 * 
	 * @param data
	 */
	void setUnlabeledData(Iterable<T> data);
	
	/**
	 * get the unlabeled data of this container.
	 * 
	 * @return unlabeled data
	 */
	Collection<T> getUnlabeledData();
	
	/**
	 * returns whether the data set solely contains factor labels.
	 * 
	 * @return 
	 */
	boolean isFactorLabelled();
}
