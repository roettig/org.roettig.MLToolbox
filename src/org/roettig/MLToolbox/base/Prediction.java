package org.roettig.MLToolbox.base;

import org.roettig.MLToolbox.base.instance.Instance;
import org.roettig.MLToolbox.base.label.Label;

public class Prediction
{
	private Instance inst;
	private Label    y;
	private Label    yp;
	
	public Prediction(Instance inst, Label y, Label yp)
	{
		this.inst = inst;
		this.y    = y;
		this.yp   = yp;
	}

	public Instance getInst()
	{
		return inst;
	}

	public Label getTrueLabel()
	{
		return y;
	}

	public Label getPredictedLabel()
	{
		return yp;
	}	
}
