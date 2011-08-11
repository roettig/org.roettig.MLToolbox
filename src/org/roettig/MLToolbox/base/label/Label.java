package org.roettig.MLToolbox.base.label;

/**
 * Abstract base class for all labels within the MLToolbox.
 * 
 * @author roettig
 *
 */
public abstract class Label
{
	/**
	 * return the numerical representation of this label, since
	 * most ML algorithms encode labels into numerical values
	 * 
	 * @return numerical representation
	 */
	public abstract double getDoubleValue();
}
