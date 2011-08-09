package org.roettig.MLToolbox.base.instance;

public interface InstanceFilter<T extends Instance>
{
	boolean accept(T t);
}
