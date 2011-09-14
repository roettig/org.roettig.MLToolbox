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
 * The CSVCModel is a classification model with parameter <i>C</i> (trade-off between training error and generalization).
 * 
 * @author roettig
 *
 * @param <T> type parameter of instances
 */
public class CSVCModel<T extends Instance> extends SVMModel<T> implements ClassificationModel
{
	private static final long	serialVersionUID	= -2300813589119920238L;

	public static String C = CSVCModel.class.getCanonicalName()+"_C";
	
	private Parameter<Double> c = new Parameter<Double>(C,new Double[]{1.0});
	
	/**
	 * ctor with kernel function to use.
	 * 
	 * @param k_fun_ kernel function
	 */
	public CSVCModel(KernelFunction<T> k_fun_)
	{
		super(k_fun_);
		this.registerParameter(C, c);
		qm = new FMeasure();
	}
		
	/**
	 * set allowed values of parameter C.
	 * 
	 * @param Cs allowed values
	 */
	public void setC(Double... Cs)
	{
		c = new Parameter<Double>(C,Cs);
		this.registerParameter(C, c);
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
		double c = (Double) this.getParameter(C).getCurrentValue();
		ret.setValue(SVMTrainingParams.C, c);
		return ret;
	}

}
