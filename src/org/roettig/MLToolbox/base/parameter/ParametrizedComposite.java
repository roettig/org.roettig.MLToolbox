package org.roettig.MLToolbox.base.parameter;

import java.util.List;

public interface ParametrizedComposite extends Parametrized	
{
	void add(Parametrized comp);
	void remove(Parametrized comp);
	List<Parametrized> getChildren();
}

