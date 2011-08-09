package org.roettig.MLToolbox.model;

import java.util.List;

import org.roettig.MLToolbox.base.Prediction;
import org.roettig.MLToolbox.base.instance.Instance;

public class SelectedModel<T extends Instance>
{
	public Model<T> model;
	public List<Prediction> predictions;
	public double qual;
	
	public SelectedModel(Model<T> model, double qual)
	{
		this.model = model;
		//this.predictions = preds;
		this.qual = qual;
	}
}
