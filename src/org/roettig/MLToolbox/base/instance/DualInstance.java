package org.roettig.MLToolbox.base.instance;

import org.roettig.MLToolbox.base.label.Label;

/**
 * DualInstance is a generic class derived from Instance parametrized by the payload type (i.e. String).
 * 
 * @author roettig
 *
 * @param <T> the type of the payload (i.e. String, ...)
 */
public final class DualInstance<T> extends Instance
{

	private final T payload; 
	
	public DualInstance(Label label, T payload)
	{
		super(label);
		this.payload = payload;
	}
	
	/**
	 * returns the payload.
	 * 
	 * @return payload
	 */
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
