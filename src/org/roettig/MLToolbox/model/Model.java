package org.roettig.MLToolbox.model;


import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
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
	
	/**
	 * returns current quality measure.
	 * 
	 * @return quality measure
	 */
	public QualityMeasure getQualityMeasure()
	{
		return this.qm;
	}

	protected QualityMeasure qm;
	
	protected InstanceContainer<T> trainingdata;
	
	public void store(String filename) throws Exception
	{
		FileOutputStream f_out = new FileOutputStream(filename);
		// Write object with ObjectOutputStream
		ObjectOutputStream obj_out = new ObjectOutputStream (f_out);
		// Write object out to disk
		obj_out.writeObject( this );
		f_out.close();
	}

	public static Model<?> load(String filename) throws Exception
	{
		// restore object from file ...
		// Read from disk using FileInputStream
		FileInputStream f_in = new FileInputStream(filename);

		// Read object using ObjectInputStream
		ObjectInputStream obj_in = new ObjectInputStream (f_in);
		Model<?> ret = (Model<?>) obj_in.readObject();
		f_in.close();
		return ret;
	}
}
