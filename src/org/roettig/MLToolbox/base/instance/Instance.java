package org.roettig.MLToolbox.base.instance;

import java.io.Serializable;
import java.util.List;

import org.roettig.MLToolbox.base.Annotated;
import org.roettig.MLToolbox.base.impl.DefaultAnnotated;
import org.roettig.MLToolbox.base.label.Label;


/**
 * Abstract base class of all Instances within the MLToolbox.   
 *   
*/
public abstract class Instance implements Annotated, Cloneable, Serializable
{
	private final Label label;
	private final long  id;
	private final Annotated props;
	
	private static long ID_COUNTER = 0;
	
	public Instance(Label label)
	{
		this.label = label;
		this.id    = ID_COUNTER++;
		this.props = new DefaultAnnotated();
	}
	
	/**
	 * returns a copy of the instance with newly assigned label.
	 * 
	 * @param lab
	 * 
	 * @return instance
	 */
	public abstract Instance reassign(Label lab);
	
	/**
	 * returns the label of the instance.
	 * 
	 * @return label
	 */
	public Label getLabel()
	{
		return label;
	}	
	
	/**
	 * returns a unique identifier of the instance.
	 * 
	 * @return id
	 */
	public long getId()
	{
		return id;
	}

	@Override
	public void addProperty(String name, Object obj)
	{
		props.addProperty(name, obj);
	}

	@Override
	public Object getProperty(String name)
	{
		return props.getProperty(name);
	}

	@Override
	public void removeProperty(String name)
	{
		props.removeProperty(name);
	}

	@Override
	public boolean hasProperty(String name)
	{
		return props.hasProperty(name);
	}

	@Override
	public List<String> getPropertyNames()
	{
		return props.getPropertyNames();
	}
}
