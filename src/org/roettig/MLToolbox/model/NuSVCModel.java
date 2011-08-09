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
import org.roettig.MLToolbox.kernels.KernelMatrix;
import org.roettig.MLToolbox.util.libsvm.libsvmDelegate;
import org.roettig.MLToolbox.util.libsvm.svm;
import org.roettig.MLToolbox.util.libsvm.svm_parameter;
import org.roettig.MLToolbox.validation.FMeasure;

public class NuSVCModel<T extends Instance>  extends LibsvmModel<T> implements ClassificationModel
{
	private static final long	serialVersionUID	= -2300813589119920238L;

	public static String NU = CSVCModel.class.getCanonicalName()+"_NU";
	
	private Parameter<Double> nu = new Parameter<Double>(NU,new Double[]{0.2});
	
	public NuSVCModel(KernelFunction<T> k_fun_)
	{
		super(k_fun_);
		this.registerParameter(NU, nu);
		qm = new FMeasure();
	}
	
	@Override
	protected void initLibsvm()
	{
		super.initLibsvm();
		libsvmdelegate.param.svm_type    = svm_parameter.NU_SVC;
		libsvmdelegate.param.kernel_type = svm_parameter.PRECOMPUTED;
		libsvmdelegate.param.nu  = (Double) this.getParameter(NU).getCurrentValue();
	}
		
	public void setNu(Double... NUs)
	{
		nu = new Parameter<Double>(NU,NUs);
		this.registerParameter(NU, nu);
	}
	
	@Override
	public void train(InstanceContainer<T> trainingdata)
	{
		this.trainingdata = trainingdata;
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
			double yp = svm.svm_predict(libsvmdelegate.model, libsvmDelegate.makeSVMnode(K.getRow(i),0));
			Label  pred_label = FactorLabel.fromDoubleValue(yp);
			Prediction pred = new Prediction(testdata.get(i),testdata.get(i).getLabel(),pred_label);
			preds.add(pred);
		}
		return preds;
	}
}
