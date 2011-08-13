/**
 * 
 */
package org.roettig.MLToolbox.validation;

import java.io.Serializable;
import java.util.Collection;

import org.roettig.MLToolbox.base.Prediction;


/**
 * QualitMeasure is the base interface for all quality measures that calculate the quality on a 
 * collection of predictions.
 *  
 * @author roettig
 *
 */
public interface QualityMeasure extends Serializable
{
	/**
	 * calculates the quality of prediction on a collection of predictions (the higher the better).
	 * 
	 * @param predictions
	 * 
	 * @return quality
	 */
    public double getQuality(Collection<Prediction> predictions);
}
