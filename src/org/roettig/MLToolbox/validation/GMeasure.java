package org.roettig.MLToolbox.validation;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.roettig.MLToolbox.base.Prediction;
import org.roettig.MLToolbox.base.label.FactorLabel;
import org.roettig.MLToolbox.base.label.Label;
import org.roettig.maths.statistics.Statistics;

/**
 * The GMeasure class gives the Gmeasure (mean of sensitivity and specificity) of predictions.
 *
 * @author roettig
 *
 */
public class GMeasure implements QualityMeasure
{

	private FactorLabel pos_label;
	
	public GMeasure()
	{
		
	}
	
	/**
	 * ctor which specifies the label to be used for the positive class.
	 * 
	 * @param pos_label
	 */
	public GMeasure(FactorLabel pos_label)
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
		
		List<Double> sens_spec = Statistics.calcSensSpec(Yp, Yt);
		
		double sens = sens_spec.get(0);
		double spec = sens_spec.get(1);
		
		return 0.5*(sens+spec);
	}
}