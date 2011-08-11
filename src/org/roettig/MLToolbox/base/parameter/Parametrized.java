package org.roettig.MLToolbox.base.parameter;

import java.util.List;


/**
 * The interface Parametrized defines the methods to register and retrieve all
 * registered Parameters.
 * 
 * @author roettig
 *
 */
public interface Parametrized
{
	/**
	 * returns all registered parameters (even from attached children) 
	 * 
	 * @return list of parameters
	 */
	List<Parameter<?>> getParameters();
	
	/**
	 * registers a parameter in this node.
	 * 
	 * @param name parameter name
	 * @param param parameter to register
	 */
	void registerParameter(String name, Parameter<?> param);
}
