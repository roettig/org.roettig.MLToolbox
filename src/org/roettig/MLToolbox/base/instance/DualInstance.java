package org.roettig.MLToolbox.base.instance;

import org.roettig.MLToolbox.base.label.Label;

public final class DualInstance<T> extends Instance
{

	private final T payload; 
	
	public DualInstance(Label label, T payload)
	{
		super(label);
		this.payload = payload;
	}
	
	public T getPayload()
	{
		return this.payload;
	}

	@Override
	public DualInstance<T> reassign(Label lab)
	{
		return new DualInstance<T>(lab,this.payload);
	}

}
