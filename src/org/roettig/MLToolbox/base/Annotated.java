package org.roettig.MLToolbox.base;

import java.util.List;

/**
 * The interface Annotated has to be implemented by classes
 * that want to store property objects.  
 */
public interface Annotated
{
	/**
	 * adds the property object <i>obj</i> with the String <i>name</i> as key
	 *
	 * @param name
	 * @param obj
	 */
	void   addProperty(String name, Object obj);

	/**
	 * retrieves the property object stored with key <i>name</i>
	 *
	 * @param name
	 * 
	 * @return Object
	 * 
	 * @throws Exception
	 */    
	Object getProperty(String name);

	/**
	 * removes the property object with the String <i>name</i> as key
	 *
	 * @param name
	 * 
	 */
	void   removeProperty(String name);

	/**
	 * checks whether the property object with the String <i>name</i> is set
	 *
	 * @param name
	 * 
	 */
	boolean   hasProperty(String name);
	
	List<String> getPropertyNames();
}
