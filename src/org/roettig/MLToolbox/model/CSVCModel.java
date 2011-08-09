package org.roettig.MLToolbox.model;


import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.roettig.MLToolbox.base.Prediction;
import org.roettig.MLToolbox.base.impl.DefaultInstanceContainer;
import org.roettig.MLToolbox.base.instance.Instance;
import org.roettig.MLToolbox.base.instance.InstanceContainer;
import org.roettig.MLToolbox.base.instance.LabelSupplier;
import org.roettig.MLToolbox.base.instance.PrimalInstance;
import org.roettig.MLToolbox.base.label.FactorLabel;
import org.roettig.MLToolbox.base.label.Label;
import org.roettig.MLToolbox.base.label.LabelFactory;
import org.roettig.MLToolbox.base.parameter.Parameter;
import org.roettig.MLToolbox.base.parameter.Parametrized;
import org.roettig.MLToolbox.kernels.KernelFunction;
import org.roettig.MLToolbox.kernels.KernelMatrix;
import org.roettig.MLToolbox.kernels.LinearKernel;
import org.roettig.MLToolbox.kernels.RBFKernel;
import org.roettig.MLToolbox.test.data.DataSource;
import org.roettig.MLToolbox.util.InstanceReader;
import org.roettig.MLToolbox.util.MLHelper;
import org.roettig.MLToolbox.util.libsvm.libsvmDelegate;
import org.roettig.MLToolbox.util.libsvm.svm;
import org.roettig.MLToolbox.util.libsvm.svm_parameter;
import org.roettig.MLToolbox.validation.AccuracyMeasure;
import org.roettig.MLToolbox.validation.FMeasure;
import org.roettig.MLToolbox.validation.GMeasure;
import org.roettig.MLToolbox.validation.ModelValidation;
import org.roettig.MLToolbox.validation.Precision;
import org.roettig.MLToolbox.validation.QualityMeasure;
import org.roettig.MLToolbox.validation.Sensitivity;
import org.roettig.MLToolbox.validation.Specificity;


public class CSVCModel<T extends Instance> extends LibsvmModel<T> implements ClassificationModel
{
	private static final long	serialVersionUID	= -2300813589119920238L;

	public static String C = CSVCModel.class.getCanonicalName()+"_C";
	
	private Parameter<Double> c = new Parameter<Double>(C,new Double[]{1.0});
	
	public CSVCModel(KernelFunction<T> k_fun_)
	{
		super(k_fun_);
		this.registerParameter(C, c);
		qm = new AccuracyMeasure();
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
		/*
		RBFKernel rbf = new RBFKernel();
		rbf.setGamma(0.00001);
		CSVCModel<PrimalInstance> m = new CSVCModel<PrimalInstance>(rbf);
		
		List<Parameter<?>> params = m.getParameters();
		for(Parameter<?> p : params)
		{
			System.out.println(p);
		}
		*/
		
		/*
		RBFKernel rbf = new RBFKernel();
		rbf.setGamma(0.01);
		
		Label lab1 = new FactorLabel("class1");
		Label lab2 = new FactorLabel("class2");
		
		PrimalInstance pi1 = new PrimalInstance(lab1,new double[]{-0.1});
		PrimalInstance pi2 = new PrimalInstance(lab1,new double[]{-0.3});
		PrimalInstance pi3 = new PrimalInstance(lab1,new double[]{-0.1});
		PrimalInstance pi4 = new PrimalInstance(lab1,new double[]{-0.3});
		PrimalInstance pi5 = new PrimalInstance(lab2,new double[]{0.1});
		PrimalInstance pi6 = new PrimalInstance(lab2,new double[]{0.3});
		PrimalInstance pi7 = new PrimalInstance(lab2,new double[]{0.1});
		PrimalInstance pi8 = new PrimalInstance(lab2,new double[]{0.3});
		
		InstanceView<PrimalInstance> iv = new InstanceView<PrimalInstance>();
		iv.add(pi1);
		iv.add(pi2);
		iv.add(pi3);
		iv.add(pi4);
		iv.add(pi5);
		iv.add(pi6);
		iv.add(pi7);
		iv.add(pi8);
		
		
		CSVCModel<PrimalInstance> m = new CSVCModel<PrimalInstance>(rbf);
		m.train(iv);
		List<Prediction> preds = m.predict(iv);
		m.getQuality(preds);
		
		double qual = ModelValidation.CV(2, iv, m);
		System.out.println(qual);
		*/
		
		RBFKernel rbf = new RBFKernel();
		rbf.setGamma(1.0,0.1,0.01,0.001);
		
		LinearKernel lK = new LinearKernel();
		
		DefaultInstanceContainer<PrimalInstance>  data = InstanceReader.read(DataSource.class.getResourceAsStream("iris.dat"), 5, true);

		CSVCModel<PrimalInstance>                 m  = new CSVCModel<PrimalInstance>(rbf);
		m.setQualityMeasure(new GMeasure());
		
		m.setC(1.0,100.0);
		
		ModelValidation.turnLoggingOff();
		
		SelectedModel<PrimalInstance> sm = ModelValidation.SimpleNestedCV(5, 5, data, m);
		QualityMeasure sens = new Sensitivity();
		QualityMeasure spec = new Specificity();
		QualityMeasure prec = new Precision();
		System.out.println("qual="+sm.qual+" sens="+sens.getQuality(sm.predictions)+" spec="+spec.getQuality(sm.predictions)+" prec="+prec.getQuality(sm.predictions));

		/*
		DefaultInstanceContainer<PrimalInstance>  train = new DefaultInstanceContainer<PrimalInstance>();
		DefaultInstanceContainer<PrimalInstance>  test  = new DefaultInstanceContainer<PrimalInstance>();
		
		ModelValidation.getStratifiedSplit(0.5, data, train, test);
		
		MLHelper.exportLIBSVM(train, "/tmp/iris.trn");
		MLHelper.exportLIBSVM(test, "/tmp/iris.tst");
		
		SelectedModel sm = ModelValidation.ModelSelection(2, train, m);
		
		//m.train(train);
		
		List<Prediction> preds = sm.model.predict(test);
		System.out.println(m.getQuality(preds));
		*/
		/*
		ModelValidation.turnLoggingOff();
		double qual = ModelValidation.CV(5, iv, m);
		System.out.println(qual);
		Random rng = new Random(2204);
		iv.shuffle(rng);
		
		qual = ModelValidation.CV(5, iv, m);
		System.out.println(qual);
		*/
		//ModelValidation.ModelSelection(2, data, m);
	}
	
}
