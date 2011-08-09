package org.roettig.MLToolbox.model;

import java.util.List;

import org.roettig.MLToolbox.base.Prediction;
import org.roettig.MLToolbox.base.instance.Instance;
import org.roettig.MLToolbox.base.instance.InstanceContainer;
import org.roettig.MLToolbox.kernels.KernelFunction;
import org.roettig.MLToolbox.util.libsvm.libsvmDelegate;
import org.roettig.MLToolbox.validation.QualityMeasure;

public abstract class LibsvmModel<T extends Instance> extends Model<T>
{
	private static final long	serialVersionUID	= 1928980059846902726L;
	
	protected libsvmDelegate<T> libsvmdelegate;
	protected KernelFunction<T> k_fun;
	protected transient InstanceContainer<T> trainingdata;
	protected QualityMeasure qm;
	
	public LibsvmModel(KernelFunction<T> k_fun_)
	{
		this.k_fun = k_fun_;
		add(k_fun);
	}
	
	protected void initLibsvm()
	{
		libsvmdelegate = new libsvmDelegate<T>(k_fun);
	}
	
	public double getObjectiveValue(int i)
	{
		return libsvmdelegate.model.obj[i];
	}
	
	@Override
	public void train(InstanceContainer<T> trainingdata)
	{
		initLibsvm();
		libsvmdelegate.train(this.trainingdata);
	}

	@Override
	public abstract List<Prediction> predict(InstanceContainer<T> testdata);

	@Override
	public double getQuality(List<Prediction> predictions)
	{
		return qm.getQuality(predictions);
	}

}
