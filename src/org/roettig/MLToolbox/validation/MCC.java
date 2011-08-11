package org.roettig.MLToolbox.validation;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.roettig.MLToolbox.base.Prediction;
import org.roettig.MLToolbox.base.label.FactorLabel;
import org.roettig.MLToolbox.base.label.Label;
import org.roettig.maths.statistics.Statistics;
import org.roettig.maths.statistics.Statistics.ConfusionMatrix;

/**
 * The MCC class gives the Matthews Correlation Coefficient of predictions.
 *
 * @author roettig
 *
 */
public class MCC implements QualityMeasure
{

	private FactorLabel pos_label;
	
	public MCC()
	{
		
	}
	
	/**
	 * ctor which specifies the label to be used for the positive class.
	 * 
	 * @param pos_label
	 */
	public MCC(FactorLabel pos_label)
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
		
		ConfusionMatrix conf = null;
		
		if(pos_label!=null)
			conf = Statistics.calcConfusionMatrix(pos_label, Yp, Yt);
		else
			conf = Statistics.calcConfusionMatrix(Yt.get(0), Yp, Yt);
		double TP = conf.TP;
		double TN = conf.TN;
		double FP = conf.FP;
		double FN = conf.FN;
		double mcc = 0.0;
		if( (TP+FP)>0 && (TP+FN)>0 && (TN+FP)>0 && (TN+FN)>0 )
		{
			mcc = (TP*TN-FP*FN)/Math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));			
		}
		return mcc;
	}

}
