package org.roettig.MLToolbox.base.parameter;

import java.util.List;

/**
 * The interface ParametrizedComposite defines the operations to built tree structures of
 * composites.
 * 
 * @author roettig
 *
 */
public interface ParametrizedComposite extends Parametrized	
{
	/**
	 * add a Parametrized object to this node.
	 * 
	 * @param comp
	 */
	void add(Parametrized comp);
	
	/**
	 * remove a Parametrized object from this node.
	 * 
	 * @param comp parameter to remove
	 */
	void remove(Parametrized comp);
	
	/**
	 * returns a list of all added Parametrized objects.
	 * 
	 * @return list of Parametrized objects
	 */
	List<Parametrized> getChildren();
}

