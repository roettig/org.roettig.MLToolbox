package org.roettig.MLToolbox.validation;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.roettig.MLToolbox.base.Prediction;
import org.roettig.MLToolbox.base.label.FactorLabel;
import org.roettig.MLToolbox.base.label.Label;
import org.roettig.maths.statistics.Statistics;

/**
 * The Precision class gives the Precision of predictions.
 *
 * @author roettig
 *
 */
public class Precision implements QualityMeasure
{

	private FactorLabel pos_label;
	
	public Precision()
	{
		
	}
	
	/**
	 * ctor which specifies the label to be used for the positive class.
	 * 
	 * @param pos_label
	 */
	public Precision(FactorLabel pos_label)
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
		List<Double> prec_rec = null;
		
		if(pos_label!=null)
			prec_rec = Statistics.calcPrecRec(pos_label, Yp, Yt);
		else
			prec_rec = Statistics.calcPrecRec(Yp, Yt);
		
		return prec_rec.get(0);
	}
}