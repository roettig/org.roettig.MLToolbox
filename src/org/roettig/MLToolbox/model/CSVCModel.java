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

public class CSVCModel<T extends Instance> extends LibsvmModel<T> implements ClassificationModel
{
	private static final long	serialVersionUID	= -2300813589119920238L;

	public static String C = CSVCModel.class.getCanonicalName()+"_C";
	
	private Parameter<Double> c = new Parameter<Double>(C,new Double[]{1.0});
	
	public CSVCModel(KernelFunction<T> k_fun_)
	{
		super(k_fun_);
		this.registerParameter(C, c);
		qm = new FMeasure();
	}
	
	@Override
	protected void initLibsvm()
	{
		super.initLibsvm();
		libsvmdelegate.param.svm_type    = svm_parameter.C_SVC;
		libsvmdelegate.param.kernel_type = svm_parameter.PRECOMPUTED;
		libsvmdelegate.param.C  = (Double) this.getParameter(C).getCurrentValue();
	}
		
	public void setC(Double... Cs)
	{
		c = new Parameter<Double>(C,Cs);
		this.registerParameter(C, c);
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
