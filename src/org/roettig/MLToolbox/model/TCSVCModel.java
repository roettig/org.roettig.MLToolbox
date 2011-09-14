package org.roettig.MLToolbox.model;

import java.util.ArrayList;
import java.util.List;

import org.roettig.MLToolbox.base.Prediction;
import org.roettig.MLToolbox.base.instance.InstanceContainer;
import org.roettig.MLToolbox.base.instance.PrimalInstance;
import org.roettig.MLToolbox.base.label.FactorLabel;
import org.roettig.MLToolbox.base.parameter.Parameter;
import org.roettig.MLToolbox.kernels.KernelFunction;
import org.roettig.MLToolbox.kernels.PolyKernel;
import org.roettig.MLToolbox.kernels.RBFKernel;
import org.roettig.MLToolbox.model.svm.DefaultSVMTrainingParams;
import org.roettig.MLToolbox.model.svm.SVMModel;
import org.roettig.MLToolbox.model.svm.SVMTrainingParams;
import org.roettig.MLToolbox.model.svm.SVMlightDelegate;
import org.roettig.MLToolbox.validation.FMeasure;

/**
 * TCSVCModel is a transductive CSVC model using the RBF or poly kernel. It is based on svmlight.
 *   
 * @author roettig
 *
 */
public class TCSVCModel extends SVMModel<PrimalInstance> implements ClassificationModel
{
	public static String C     = TCSVCModel.class.getCanonicalName()+"_C";
	public static String GAMMA = TCSVCModel.class.getCanonicalName()+"_gamma";
	public static String DEG   = TCSVCModel.class.getCanonicalName()+"_degree";
	
	private Parameter<Double>  c = new Parameter<Double>(C,new Double[]{1.0});
	private Parameter<Double>  g = new Parameter<Double>(GAMMA,new Double[]{1.0});
	private Parameter<Integer> d = new Parameter<Integer>(DEG,new Integer[]{2});
	
	public static FactorLabel POS = new FactorLabel("pos",1.0);
	public static FactorLabel NEG = new FactorLabel("neg",-1.0);
	
	protected boolean transductive = false;
	
	public TCSVCModel(KernelFunction<PrimalInstance> k_fun)
	{
		super(k_fun);
		this.registerParameter(C, c);
		if(k_fun instanceof RBFKernel)
			this.registerParameter(GAMMA, g);
		if(k_fun instanceof PolyKernel)
			this.registerParameter(DEG, d);
		qm = new FMeasure();
	}
	
	/**
	 * set allowed values for parameter C.
	 * 
	 * @param Cs
	 */
	public void setC(Double... Cs)
	{
		c = new Parameter<Double>(C,Cs);
		this.registerParameter(C, c);
	}
	
	/**
	 * set allowed values for parameter gamma (RBFKernel).
	 * 
	 * @param Gs
	 */
	public void setGamma(Double... Gs)
	{
		g = new Parameter<Double>(GAMMA,Gs);
		this.registerParameter(GAMMA, g);
	}
	
	/**
	 * set allowed values for parameter deg (PolyKernel).
	 * 
	 * @param Gs
	 */
	public void setDegree(Integer... Ds)
	{
		d = new Parameter<Integer>(DEG,Ds);
		this.registerParameter(DEG, d);
	}
	
	public void transductiveMode(boolean flag)
	{
		transductive = flag;
	}
			
	@Override
	public void train(InstanceContainer<PrimalInstance> trainingdata)
	{
		delegate = new SVMlightDelegate();
		super.train(trainingdata);
	}
	
	@Override
	public List<Prediction> predict(InstanceContainer<PrimalInstance> testdata)
	{
		List<Double> yps = delegate.predict(testdata);
		List<Prediction> ret = new ArrayList<Prediction>();
		int idx = 0;
		for(Double yp: yps)
		{
			if(yp>=0)
			{
				Prediction p = new Prediction(testdata.get(idx),testdata.get(idx).getLabel(),POS);
				ret.add(p);
			}
			else
			{
				Prediction p = new Prediction(testdata.get(idx),testdata.get(idx).getLabel(),NEG);
				ret.add(p);
			}
			idx++;
		}
		return ret;
	}

	@Override
	protected SVMTrainingParams getTrainingParameters()
	{
		SVMTrainingParams ret = new DefaultSVMTrainingParams();
		double c = (Double) this.getParameter(C).getCurrentValue();
		ret.setValue(SVMTrainingParams.C, c);
		if(transductive)
			ret.setFlag(SVMTrainingParams.Transductive);
		return ret;
	}
}
