package org.roettig.MLToolbox.model;

import org.roettig.MLToolbox.base.instance.Instance;
import org.roettig.MLToolbox.base.instance.InstanceContainer;
import org.roettig.MLToolbox.base.parameter.Parameter;
import org.roettig.MLToolbox.kernels.KernelFunction;
import org.roettig.MLToolbox.model.svm.DefaultSVMTrainingParams;
import org.roettig.MLToolbox.model.svm.LIBSVMDelegate;
import org.roettig.MLToolbox.model.svm.SVMModel;
import org.roettig.MLToolbox.model.svm.SVMTrainingParams;
import org.roettig.MLToolbox.validation.FMeasure;

/**
 * The NuSVCModel is a classification model with parameter <i>nu</i> (upper bound on support vectors).
 * 
 * @author roettig
 *
 * @param <T> type parameter of instances
 */
public class NuSVCModel<T extends Instance>  extends SVMModel<T> implements ClassificationModel
{
	private static final long	serialVersionUID	= -2300813589119920238L;

	public static String NU = CSVCModel.class.getCanonicalName()+"_NU";
	
	private Parameter<Double> nu = new Parameter<Double>(NU,new Double[]{0.2});
	
	/**
	 * ctor with kernel function to use.
	 * 
	 * @param k_fun_ kernel function
	 */
	public NuSVCModel(KernelFunction<T> k_fun_)
	{
		super(k_fun_);
		this.registerParameter(NU, nu);
		qm = new FMeasure();
	}
	
	/**
	 * set allowed values for parameter nu.
	 * 
	 * @param NUs
	 */
	public void setNu(Double... NUs)
	{
		nu = new Parameter<Double>(NU,NUs);
		this.registerParameter(NU, nu);
	}
		
	@Override
	public void train(InstanceContainer<T> trainingdata)
	{
		delegate = new LIBSVMDelegate<T>();
		super.train(trainingdata);
	}

	@Override
	protected SVMTrainingParams getTrainingParameters()
	{
		SVMTrainingParams ret = new DefaultSVMTrainingParams();
		double nu = (Double) this.getParameter(NU).getCurrentValue();
		ret.setValue(SVMTrainingParams.NU, nu);
		return ret;
	}
}
