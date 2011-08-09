package org.roettig.MLToolbox.base.parameter;

import java.util.List;



public interface Parametrized
{
	List<Parameter<?>> getParameters();
	void registerParameter(String name, Parameter<?> param);
}
