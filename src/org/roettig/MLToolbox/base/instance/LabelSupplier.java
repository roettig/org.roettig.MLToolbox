package org.roettig.MLToolbox.base.instance;

import org.roettig.MLToolbox.base.label.Label;

public interface LabelSupplier
{
	Label getLabel(int idx);
}
