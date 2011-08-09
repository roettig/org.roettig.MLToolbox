package org.roettig.MLToolbox.model;


import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.roettig.MLToolbox.base.Prediction;
import org.roettig.MLToolbox.base.impl.DefaultInstanceContainer;
import org.roettig.MLToolbox.base.instance.Instance;
import org.roettig.MLToolbox.base.instance.InstanceContainer;
import org.roettig.MLToolbox.base.instance.PrimalInstance;
import org.roettig.MLToolbox.base.label.Label;
import org.roettig.MLToolbox.base.label.NumericLabel;
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
import org.roettig.MLToolbox.validation.PearsonMeasure;
import org.roettig.MLToolbox.validation.QualityMeasure;


public class NuSVRModel<T extends Instance> extends LibsvmModel<T>
{
	private static final long	serialVersionUID	= 6206408147990280932L;
	
	public static String NU = NuSVRModel.class.getCanonicalName()+"_NU";
	public static String C  = NuSVRModel.class.getCanonicalName()+"_C";
	
	private Parameter<Double> nu = new Parameter<Double>(NU,new Double[]{0.5});
	private Parameter<Double> c  = new Parameter<Double>(C,new Double[]{1.0});

	public NuSVRModel(KernelFunction<T> k_fun_)
	{
		super(k_fun_);
		
		qm = new PearsonMeasure();
		
		this.registerParameter(NU, nu);
		this.registerParameter(C, c);
	}
	
	@Override
	protected void initLibsvm()
	{
		super.initLibsvm();
		libsvmdelegate.param.svm_type    = svm_parameter.NU_SVR;
		libsvmdelegate.param.kernel_type = svm_parameter.PRECOMPUTED;
		libsvmdelegate.param.C  = (Double) this.getParameter(C).getCurrentValue();
		libsvmdelegate.param.nu = (Double) this.getParameter(NU).getCurrentValue();
	}
	
	public void setC(Double... Cs)
	{
		Parameter<Double> c = new Parameter<Double>(C,Cs);
		this.registerParameter(C, c);
	}
	
	public void setNU(Double... NUs)
	{
		Parameter<Double> nu = new Parameter<Double>(NU,NUs);
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

		KernelMatrix K = KernelMatrix.compute(trainingdata, testdata, k_fun);

		int Nt = testdata.size();
		for(int i=0;i<Nt;i++)
		{
			double yp = svm.svm_predict(libsvmdelegate.model, libsvmDelegate.makeSVMnode(K.getRow(i),0));
			Label  pred_label = new NumericLabel(yp);
			Prediction pred = new Prediction(testdata.get(i),testdata.get(i).getLabel(),pred_label);
			preds.add(pred);
		}
		return preds;
	}

	@Override
	public double getQuality(List<Prediction> predictions)
	{
		return qm.getQuality(predictions);
	}
	
	@Override
	public void setQualityMeasure(QualityMeasure qm)
	{
		this.qm = qm;
	}
	
	public static void main(String[] args) throws Exception
	{

		RBFKernel rbf = new RBFKernel();
		rbf.setGamma(1.0);
		
		DefaultInstanceContainer<PrimalInstance> samples = InstanceReader.read(DataSource.class.getResourceAsStream("sin.dat"), 2, false);
		
		NuSVRModel<PrimalInstance>     m  = new NuSVRModel<PrimalInstance>(rbf);
		m.setC(10.0);
		m.setNU(0.2);
		
		DefaultInstanceContainer<PrimalInstance>  train = new DefaultInstanceContainer<PrimalInstance>();
		DefaultInstanceContainer<PrimalInstance>  test  = new DefaultInstanceContainer<PrimalInstance>();
		
		samples.shuffle(new Random(2204));
		
		ModelValidation.getSplit(0.8, samples, train, test);
		
		MLHelper.exportLIBSVM(train, "/tmp/sin.trn");
		MLHelper.exportLIBSVM(test, "/tmp/sin.tst");
		
		m.train(train);
		List<Prediction> preds = m.predict(test);
		System.out.println(m.getQuality(preds));
		
		/*
		List<Prediction> preds = new ArrayList<Prediction>();
		double qual = ModelValidation.CV(5, samples, m, preds);
		
		for(Prediction pred: preds)
		{
			System.out.println(pred.getTrueLabel().getDoubleValue()+" "+pred.getPredictedLabel().getDoubleValue());
		}
		
		System.out.println(qual);
		*/
	}
}
