package org.roettig.MLToolbox.base;

import java.util.List;

/**
 * The interface Annotated has to be implemented by classes
 * that want to store property objects.
 * 
 * <br/>
 * <br/>
 * 
 * Note: For ease of use simply delegate implementations of this interface
 *       to a DefaultAnnotated object.
 * 
 */
public interface Annotated
{
	/**
	 * adds the property object <i>obj</i> with the String <i>name</i> as key
	 *
	 * @param name key of the property
	 * @param obj  the value of the property
	 */
	void   addProperty(String name, Object obj);

	/**
	 * retrieves the property object stored with key <i>name</i>
	 *
	 * @param name key of the property
	 * 
	 * @return Object
	 * 
	 */    
	Object getProperty(String name);

	/**
	 * removes the property object with the String <i>name</i> as key
	 *
	 * @param name key of the property
	 * 
	 */
	void   removeProperty(String name);

	/**
	 * checks whether the property object with the String <i>name</i> is set
	 *
	 * @param name key of the property
	 * 
	 */
	boolean   hasProperty(String name);
	
	/**
	 * returns a List of all stored property names.
	 * 
	 * @return list of property names
	 */
	List<String> getPropertyNames();
}
