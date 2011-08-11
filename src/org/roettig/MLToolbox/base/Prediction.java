package org.roettig.MLToolbox.base;

import org.roettig.MLToolbox.base.instance.Instance;
import org.roettig.MLToolbox.base.label.Label;

/**
 * 
 * The Prediction class encapsulates pairs of predicted and true label, along with the input instance.
 * 
 * @author roettig
 *
 */
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

	/**
	 * return the instance that was the targeg of the prediction.
	 * 
	 * @return instance
	 */
	public Instance getInst()
	{
		return inst;
	}

	/**
	 * returns the true label.
	 * 
	 * @return true label
	 */
	public Label getTrueLabel()
	{
		return y;
	}

	/**
	 * returns the predicted label.
	 * 
	 * @return predicted label
	 */
	public Label getPredictedLabel()
	{
		return yp;
	}	
}
