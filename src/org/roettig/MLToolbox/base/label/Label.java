package org.roettig.MLToolbox.base.label;

import java.io.Serializable;

/**
 * Abstract base class for all labels within the MLToolbox.
 * 
 * @author roettig
 *
 */
public abstract class Label implements Serializable
{
	/**
	 * return the numerical representation of this label, since
	 * most ML algorithms encode labels into numerical values
	 * 
	 * @return numerical representation
	 */
	public abstract double getDoubleValue();
}
