package org.roettig.MLToolbox.model;


import java.util.List;
import org.roettig.MLToolbox.base.Prediction;
import org.roettig.MLToolbox.base.impl.DefaultParametrizedComposite;
import org.roettig.MLToolbox.base.instance.Instance;
import org.roettig.MLToolbox.base.instance.InstanceContainer;
import org.roettig.MLToolbox.validation.QualityMeasure;

/**
 * Abstract base class for all models within the MLToolbox.
 * 
 * @author roettig
 *
 * @param <T> : any subclass of Instance (i.e. PrimalInstance or DualInstance<S>) 
 */
public abstract class Model<T extends Instance> extends DefaultParametrizedComposite
{
	/**
	 * train model on supplied training data.
	 * 
	 * @param trainingdata
	 */
	public abstract void train(InstanceContainer<T> trainingdata);
	
	/**
	 * apply (already trained) model on supplied test data.
	 * 
	 * @param testdata
	 * 
	 * @return list of predictions
	 */
	public abstract List<Prediction> predict(InstanceContainer<T> testdata);
	
	/**
	 * calculates the quality of a supplied list of predictions.
	 * 
	 * @param predictions
	 * 
	 * @return quality of prediction (the higher the better).
	 * 
	 */
	public double getQuality(List<Prediction> predictions)
	{
		return qm.getQuality(predictions);
	}
	
	/**
	 * change the quality measure of the model.
	 * 
	 * @param qm
	 */
	public void setQualityMeasure(QualityMeasure qm)
	{
		this.qm = qm;
	}

	protected QualityMeasure qm;
	
	protected transient InstanceContainer<T> trainingdata;
}
