package org.roettig.MLToolbox.base;

import java.util.ArrayList;
import java.util.Collection;

import org.roettig.MLToolbox.base.instance.Instance;
import org.roettig.MLToolbox.base.label.FactorLabel;
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
	
	public String toString()
	{
		if(y instanceof FactorLabel)
			return String.format("yt:%s yp:%s", (this.y!=null? this.y.toString() : "null") , (this.yp!=null? this.yp.toString() : "null"));
		else
			return String.format("yt:%e yp:%e", this.y.getDoubleValue(), this.yp.getDoubleValue());
	}
	
	public static void split(Collection<Prediction> in, Collection<Label> yt, Collection<Label> yp)
	{
		for(Prediction p: in)
		{
			yt.add(p.getTrueLabel());
			yp.add(p.getPredictedLabel());
		}
	}
}
