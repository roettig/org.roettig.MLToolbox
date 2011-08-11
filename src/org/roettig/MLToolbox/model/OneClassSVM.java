package org.roettig.MLToolbox.model;


import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

import org.roettig.MLToolbox.base.Prediction;
import org.roettig.MLToolbox.base.impl.DefaultInstanceContainer;
import org.roettig.MLToolbox.base.instance.FilteredDataView;
import org.roettig.MLToolbox.base.instance.Instance;
import org.roettig.MLToolbox.base.instance.InstanceContainer;
import org.roettig.MLToolbox.base.instance.InstanceFilter;
import org.roettig.MLToolbox.base.instance.InvertedFilter;
import org.roettig.MLToolbox.base.instance.LabelSupplier;
import org.roettig.MLToolbox.base.instance.PrimalInstance;
import org.roettig.MLToolbox.base.label.FactorLabel;
import org.roettig.MLToolbox.base.label.Label;
import org.roettig.MLToolbox.base.parameter.Parameter;
import org.roettig.MLToolbox.kernels.KernelFunction;
import org.roettig.MLToolbox.kernels.KernelMatrix;
import org.roettig.MLToolbox.kernels.RBFKernel;
import org.roettig.MLToolbox.test.data.DataSource;
import org.roettig.MLToolbox.util.InstanceReader;
import org.roettig.MLToolbox.util.MLHelper;
import org.roettig.MLToolbox.util.libsvm.libsvmDelegate;
import org.roettig.MLToolbox.util.libsvm.svm;
import org.roettig.MLToolbox.util.libsvm.svm_parameter;
import org.roettig.MLToolbox.validation.ModelValidation;
import org.roettig.MLToolbox.validation.Precision;
import org.roettig.MLToolbox.validation.QualityMeasure;
import org.roettig.MLToolbox.validation.Sensitivity;
import org.roettig.MLToolbox.validation.Specificity;

public class OneClassSVM<T extends Instance> extends LibsvmModel<T> implements ClassificationModel
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
	
	@Override
	protected void initLibsvm()
	{
		super.initLibsvm();
		libsvmdelegate.param.svm_type    = svm_parameter.ONE_CLASS;
		libsvmdelegate.param.kernel_type = svm_parameter.PRECOMPUTED;
		libsvmdelegate.param.nu  = (Double) this.getParameter(NU).getCurrentValue();
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
	
	private Label pos_label;
	
	@Override
	public void train(InstanceContainer<T> trainingdata)
	{
		this.trainingdata = trainingdata;
		if(trainingdata.getLabelSupplier()!=null)
		{
			pos_label = trainingdata.getLabelSupplier().getLabel(0); 
		}
		super.train(this.trainingdata);
	}

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
}
