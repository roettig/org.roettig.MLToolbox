package org.roettig.MLToolbox.validation;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.roettig.MLToolbox.base.Prediction;
import org.roettig.MLToolbox.base.label.FactorLabel;
import org.roettig.MLToolbox.base.label.Label;
import org.roettig.maths.statistics.Statistics;

public class FMeasure implements QualityMeasure
{

	private FactorLabel pos_label;
	
	public FMeasure()
	{
		
	}
	
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
		
		return Statistics.calcFMeasure(Yp, Yt);
	}
}