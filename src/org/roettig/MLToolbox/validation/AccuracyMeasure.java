/**
 * 
 */
package org.roettig.MLToolbox.validation;

import java.io.Serializable;
import java.util.Collection;

import org.roettig.MLToolbox.base.Prediction;


/**
 * The AccuracyMeasure class gives the accuracy of predictions.
 *
 * @author roettig
 *
 */
public class AccuracyMeasure implements QualityMeasure, Serializable
{
	private static final long serialVersionUID = -4368734671968743689L;

	@Override
	public double getQuality(Collection<Prediction> predictions)
	{
		int  err = 0;
		int  N   = predictions.size(); 
		for(Prediction pred: predictions)
		{
			if(!pred.getPredictedLabel().equals(pred.getTrueLabel()))
				err++;
		}
		return (1.0-((err*1.0)/N));
	}
}