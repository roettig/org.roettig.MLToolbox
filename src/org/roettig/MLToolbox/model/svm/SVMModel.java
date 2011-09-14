package org.roettig.MLToolbox.model.svm;

import java.util.ArrayList;
import java.util.List;

import org.roettig.MLToolbox.base.Prediction;
import org.roettig.MLToolbox.base.instance.Instance;
import org.roettig.MLToolbox.base.instance.InstanceContainer;
import org.roettig.MLToolbox.base.label.FactorLabel;
import org.roettig.MLToolbox.base.label.Label;
import org.roettig.MLToolbox.base.label.NumericLabel;
import org.roettig.MLToolbox.kernels.KernelFunction;
import org.roettig.MLToolbox.model.Model;

public abstract class SVMModel<T extends Instance> extends Model<T>
{
	protected SVMModelDelegate<T> delegate;
	protected KernelFunction<T> k_fun;
	
	public SVMModel(KernelFunction<T> k_fun)
	{
		this.k_fun = k_fun;
	}
	
	@Override
	public void train(InstanceContainer<T> trainingdata)
	{
		this.trainingdata = trainingdata;
		try
		{
			delegate.train( trainingdata, k_fun, getTrainingParameters() );
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
	}

	@Override
	public List<Prediction> predict(InstanceContainer<T> testdata)
	{
		List<Prediction> ret   = new ArrayList<Prediction>();
		List<Double>     preds = delegate.predict(testdata);
		
		int N = testdata.size();
		for(int i=0;i<N;i++)
		{
			double yp = preds.get(i); 
			Label  pred_label = null;
			
			if(trainingdata.isFactorLabelled())
				pred_label = FactorLabel.fromDoubleValue(yp);
			else
				pred_label = new NumericLabel(yp);
			
			Prediction pred = new Prediction(testdata.get(i),testdata.get(i).getLabel(),pred_label);
			ret.add(pred);
		}
		return ret;
	}

	public double getObjectiveValue(int i)
	{
		return delegate.getObjectiveValue(i);
	}	
	
	protected abstract SVMTrainingParams getTrainingParameters();
}
