package org.roettig.MLToolbox.model;

import java.util.ArrayList;
import java.util.List;

import org.roettig.MLToolbox.base.Prediction;
import org.roettig.MLToolbox.base.instance.Instance;
import org.roettig.MLToolbox.base.instance.InstanceContainer;
import org.roettig.MLToolbox.base.label.FactorLabel;
import org.roettig.MLToolbox.base.label.Label;
import org.roettig.MLToolbox.base.parameter.Parameter;
import org.roettig.MLToolbox.kernels.KernelFunction;
import org.roettig.MLToolbox.model.svm.DefaultSVMTrainingParams;
import org.roettig.MLToolbox.model.svm.LIBSVMDelegate;
import org.roettig.MLToolbox.model.svm.SVMModel;
import org.roettig.MLToolbox.model.svm.SVMTrainingParams;
import org.roettig.MLToolbox.validation.Sensitivity;


public class OneClassSVM<T extends Instance> extends SVMModel<T> implements ClassificationModel
{
	private static final long	serialVersionUID	= -9073527647916178940L;

	public static String NU = OneClassSVM.class.getCanonicalName()+"_NU";
	
	private Parameter<Double> nu = new Parameter<Double>(NU,new Double[]{0.05});
	
	public static FactorLabel pos = new FactorLabel("pos");
	public static FactorLabel neg = new FactorLabel("neg");
 
	/**
	 * ctor with kernel function to use.
	 * 
	 * @param k_fun_ kernel function
	 */
	public OneClassSVM(KernelFunction<T> k_fun_)
	{
		super(k_fun_);
		this.registerParameter(NU, nu);
		qm = new Sensitivity(pos);
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

	//private Label pos_label;
	
	/*
	@Override
	public List<Prediction> predict(InstanceContainer<T> testdata)
	{
		List<Prediction> preds = new ArrayList<Prediction>();
		
		KernelMatrix K = KernelMatrix.compute(trainingdata, testdata, k_fun, k_fun.isNormalized());
		
		int Nt = testdata.size();
		for(int i=0;i<Nt;i++)
		{
			double val[] = new double[1];
 			svm.svm_predict_values(libsvmdelegate.model, libsvmDelegate.makeSVMnode(K.getRow(i),0),val);
			
 			double yp = val[0];
			
			Label  pred_label = neg; 
			
			if(yp>0.0)
				pred_label = pos;
			
			Prediction pred = null;
			if(testdata.getLabelSupplier()!=null)
			{
				if(testdata.getLabelSupplier().getLabel(i).equals(pos_label))
					pred = new Prediction(testdata.get(i), pos, pred_label);
				else
					pred = new Prediction(testdata.get(i), neg, pred_label);
			}
			else
			{
				pred = new Prediction(testdata.get(i), pos, pred_label);
			}
			
			 
			
			preds.add(pred);
		}
		return preds;
	}
	*/
	
	
	@Override
	protected SVMTrainingParams getTrainingParameters()
	{
		SVMTrainingParams ret = new DefaultSVMTrainingParams();
		double nu = (Double) this.getParameter(NU).getCurrentValue();
		ret.setValue(SVMTrainingParams.NU, nu);
		ret.setFlag(SVMTrainingParams.OneClass);
		return ret;
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
			if(yp>=0)
				pred_label = pos;
			else
				pred_label = neg;
			Prediction pred = new Prediction(testdata.get(i), pos, pred_label);
			//Prediction pred = new Prediction(testdata.get(i),testdata.get(i).getLabel(),pred_label);
			ret.add(pred);
		}
		return ret;
	}
}
