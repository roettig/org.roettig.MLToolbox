package org.roettig.MLToolbox.validation;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.roettig.MLToolbox.base.Prediction;
import org.roettig.MLToolbox.base.label.FactorLabel;
import org.roettig.MLToolbox.base.label.Label;
import org.roettig.maths.statistics.Statistics;

/**
 * The Specificity class gives the Specificity (true negative rate) of predictions.
 *
 * @author roettig
 *
 */
public class Specificity implements QualityMeasure
{

	private FactorLabel pos_label;
	
	public Specificity()
	{
		
	}
	
	/**
	 * ctor which specifies the label to be used for the positive class.
	 * 
	 * @param pos_label
	 */
	public Specificity(FactorLabel pos_label)
	{
		this.pos_label = pos_label;
	}
	
	@Override
	public double getQuality(Collection<Prediction> predictions)
	{
		List<Label> Yt = new ArrayList<Label>();
		List<Label> Yp = new ArrayList<Label>();
		for(Prediction p: predictions)
		{
			Yt.add(p.getTrueLabel());
			Yp.add(p.getPredictedLabel());
		}
		List<Double> sens_spec = null;
		if(pos_label!=null)
			sens_spec = Statistics.calcSensSpec(pos_label, Yp, Yt);
		else
			sens_spec = Statistics.calcSensSpec(Yp, Yt);
		
		return sens_spec.get(1);
	}
}