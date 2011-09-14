package org.roettig.MLToolbox.model;

import org.roettig.MLToolbox.base.instance.Instance;
import org.roettig.MLToolbox.base.instance.InstanceContainer;
import org.roettig.MLToolbox.base.parameter.Parameter;
import org.roettig.MLToolbox.kernels.KernelFunction;
import org.roettig.MLToolbox.model.svm.DefaultSVMTrainingParams;
import org.roettig.MLToolbox.model.svm.LIBSVMDelegate;
import org.roettig.MLToolbox.model.svm.SVMModel;
import org.roettig.MLToolbox.model.svm.SVMTrainingParams;
import org.roettig.MLToolbox.validation.PearsonMeasure;

/**
 * The NuSVRModel is a regression model with parameter <i>nu</i>  and <i>C</i>.
 * 
 * @author roettig
 *
 * @param <T> type parameter of instances
 */
public class NuSVRModel<T extends Instance> extends SVMModel<T>
{
	private static final long	serialVersionUID	= 6206408147990280932L;
	
	public static String NU = NuSVRModel.class.getCanonicalName()+"_NU";
	public static String C  = NuSVRModel.class.getCanonicalName()+"_C";
	
	private Parameter<Double> nu = new Parameter<Double>(NU,new Double[]{0.5});
	private Parameter<Double> c  = new Parameter<Double>(C,new Double[]{1.0});

	/**
	 * ctor with kernel function to use.
	 * 
	 * @param k_fun_ kernel function
	 */
	public NuSVRModel(KernelFunction<T> k_fun_)
	{
		super(k_fun_);
		qm = new PearsonMeasure();
		this.registerParameter(NU, nu);
		this.registerParameter(C, c);
	}
	
	/**
	 * set allowed values for parameter C.
	 * @param Cs
	 */
	public void setC(Double... Cs)
	{
		Parameter<Double> c = new Parameter<Double>(C,Cs);
		this.registerParameter(C, c);
	}

	/**
	 * set allowed values for parameter nu.
	 * 
	 * @param NUs
	 */
	public void setNU(Double... NUs)
	{
		Parameter<Double> nu = new Parameter<Double>(NU,NUs);
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
		double c = (Double) this.getParameter(C).getCurrentValue();
		ret.setValue(SVMTrainingParams.C, c);
		return ret;
	}

}
