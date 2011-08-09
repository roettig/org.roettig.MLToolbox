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
	
	public static void main(String[] args) throws Exception
	{
		
		/*
		MLHelper.exportLIBSVM(train, "/tmp/iris.trn");
		MLHelper.exportLIBSVM(test, "/tmp/iris.tst");
		
		m.train(train);
		
		List<Prediction> preds = m.predict(test);
		
		double sens = m.getQuality(preds);
		
		System.out.println("sens="+sens);
		
		preds = m.predict(train);
		
		sens = m.getQuality(preds);
		
		System.out.println("sens="+sens);
		*/
		
		
		/*
		SelectedModel<PrimalInstance> sm = ModelValidation.ModelSelection(5, train, m);
		
		System.out.println("sens(CV)="+sm.qual);
		
		List<Prediction> preds = sm.model.predict(test);
		double sens = m.getQuality(preds);
		System.out.println("sens(test)="+sens);
		*/
		//double q = ModelValidation.randomizedExternalCV(0.8, 3, 10, 2204, data, m);
		//System.out.println("q="+q);

		Random rng = new Random(2204);
		for(int i=0;i<10;i++)
		{

			DefaultInstanceContainer<PrimalInstance> sample = InstanceReader.readExt(new FileInputStream("/tmp/nrps.dat"), 1, true);

			double gam = RBFKernel.estimateGamma(sample);

			RBFKernel rbf = new RBFKernel();
			rbf.setGamma(MLHelper.makeExpSequence(-10, 10, 1, gam));

			OneClassSVM<PrimalInstance>  m  = new OneClassSVM<PrimalInstance>(rbf);

			m.setNu(0.001,0.01,0.1);

			sample.shuffle(rng);
			
			InstanceFilter<PrimalInstance> ala_filter = new InstanceFilter<PrimalInstance>(){
				@Override
				public boolean accept(PrimalInstance t)
				{
					// aad, ala, arg, asn, asp, bth, cys
					return t.getLabel().toString().equals("ala");
				}};

				InstanceFilter<PrimalInstance> not_ala_filter = new InvertedFilter<PrimalInstance>(ala_filter);

				FilteredDataView<PrimalInstance> pos = new FilteredDataView<PrimalInstance>();
				pos.addFilter(ala_filter);
				pos.addAll(sample);
				pos.setLabelSupplier(new LabelSupplier(){
					@Override
					public Label getLabel(int idx)
					{
						return OneClassSVM.pos;
					}});

				FilteredDataView<PrimalInstance> neg = new FilteredDataView<PrimalInstance>();
				neg.addFilter(not_ala_filter);
				neg.addAll(sample);
				neg.setLabelSupplier(new LabelSupplier(){
					@Override
					public Label getLabel(int idx)
					{
						return OneClassSVM.neg;
					}});

				ModelValidation.turnLoggingOff();
				SelectedModel<PrimalInstance> sm = ModelValidation.ModelSelection(3, pos, m);



				List<Prediction> neg_preds = sm.model.predict(neg);
				for(Prediction p: neg_preds)
				{
					//System.out.println(p.getPredictedLabel()+" "+p.getTrueLabel());
				}


				QualityMeasure spec = new Specificity(OneClassSVM.pos);
				//System.out.println(spec.getQuality(neg_preds));

				List<Prediction> pos_preds = sm.model.predict(pos);
				for(Prediction p: pos_preds)
				{
					//System.out.println(p.getPredictedLabel()+" "+p.getTrueLabel());
				}

				QualityMeasure prec_m = new Precision(OneClassSVM.pos);
				QualityMeasure sens_m = new Sensitivity(OneClassSVM.pos);
				double prec = prec_m.getQuality(pos_preds);
				double sens = sens_m.getQuality(pos_preds);
				double f    = 2.0*(prec*sens)/(prec+sens);
				System.out.println(prec+" "+sens+" "+f);
				//System.out.println(sens);
				//System.out.println();
		}
	}
}
