package org.roettig.MLToolbox.validation;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.roettig.MLToolbox.base.Prediction;
import org.roettig.maths.statistics.Statistics;

public class PearsonMeasure implements QualityMeasure
{

	@Override
	public double getQuality(Collection<Prediction> predictions)
	{
		List<Double> yt = new ArrayList<Double>();
		List<Double> yp = new ArrayList<Double>();
		for(Prediction pred: predictions)
		{
			yt.add(pred.getTrueLabel().getDoubleValue());
			yp.add(pred.getPredictedLabel().getDoubleValue());
		}
		return Statistics.pearsonCorr(yt, yp);
	}

}
