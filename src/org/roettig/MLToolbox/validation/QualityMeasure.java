/**
 * 
 */
package org.roettig.MLToolbox.validation;

import java.util.Collection;

import org.roettig.MLToolbox.base.Prediction;


/**
 * @author roettig
 *
 */
public interface QualityMeasure
{
    public double getQuality(Collection<Prediction> predictions);
}
