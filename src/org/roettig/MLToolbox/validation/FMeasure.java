package org.roettig.MLToolbox.validation;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.roettig.MLToolbox.base.Prediction;
import org.roettig.MLToolbox.base.label.FactorLabel;
import org.roettig.MLToolbox.base.label.Label;
import org.roettig.maths.statistics.Statistics;

/**
 * The FMeasure class gives the Fmeasure (harmonic mean of precision and recall) of predictions.
 *
 * @author roettig
 *
 */
public class FMeasure implements QualityMeasure
{

	private FactorLabel pos_label;
	
	public FMeasure()
	{	
	}
	
	/**
	 * ctor which specifies the label to be used for the positive class.
	 * 
	 * @param pos_label
	 */
	public FMeasure(FactorLabel pos_label)
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
		
		double F = 0.0;
		
		if(pos_label!=null)
			F = Statistics.calcFMeasure(pos_label, Yp, Yt);
		else
			F = Statistics.calcFMeasure(Yp, Yt);
		
		return F;
	}
}