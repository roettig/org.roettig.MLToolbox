package org.roettig.MLToolbox.base.instance;

import java.util.List;

import org.roettig.MLToolbox.base.Annotated;
import org.roettig.MLToolbox.base.impl.DefaultAnnotated;
import org.roettig.MLToolbox.base.label.Label;


/**
 * Abstract base class of all Instances within the ML4 framework.   
 *   
*/
public abstract class Instance implements Annotated
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
	
	public Label getLabel()
	{
		return label;
	}	
	
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
