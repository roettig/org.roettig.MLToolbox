package org.roettig.MLToolbox.model.svm;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import jnisvmlight.KernelParam;
import jnisvmlight.LabeledFeatureVector;
import jnisvmlight.SVMLightInterface;
import jnisvmlight.SVMLightModel;
import jnisvmlight.TrainingParameters;

import org.roettig.MLToolbox.base.instance.InstanceContainer;
import org.roettig.MLToolbox.base.instance.PrimalInstance;
import org.roettig.MLToolbox.kernels.KernelFunction;
import org.roettig.MLToolbox.kernels.LinearKernel;
import org.roettig.MLToolbox.kernels.PolyKernel;
import org.roettig.MLToolbox.kernels.RBFKernel;


public class SVMlightDelegate implements SVMModelDelegate<PrimalInstance>
{
	protected SVMLightModel      model;
	protected TrainingParameters tp = new TrainingParameters();
	
	@Override
	public void train(InstanceContainer<PrimalInstance> trainingdata, KernelFunction<PrimalInstance> k_fun, SVMTrainingParams params) throws Exception
	{
		setup( trainingdata, k_fun, params);
		
		SVMLightInterface trainer = new SVMLightInterface();

		Collection<PrimalInstance> unlab = trainingdata.getUnlabeledData();
		
		int N = trainingdata.size();
		int U = unlab.size();
		
	    
	    LabeledFeatureVector[] traindata = transform(trainingdata);
	    
	    LabeledFeatureVector[] td = null;
	    
	    if(params.hasKey(SVMTrainingParams.Transductive))
	    	td = new LabeledFeatureVector[N+U];
	    else
	    	td = new LabeledFeatureVector[N];
	    
	    for(int i=0;i<N;i++)
	    {
	    	td[i] = traindata[i];
	    }
	    if(params.hasKey(SVMTrainingParams.Transductive))
	    {
	    	int i=0;
	    	for(PrimalInstance pi: unlab)
	    	{
	    		td[N+i] = transformU(pi);
	    		i++;
	    	}
	    }
	    
	    // Switch on some debugging output
	    tp.getLearningParameters().verbosity = 0;
	    tp.getLearningParameters().svm_costratio_unlab = 10.0;
	    tp.getLearningParameters().svm_c_factor = 2.0;
	    
	    
	    tp.getLearningParameters().predfile = System.getProperty("java.io.tmpdir")+File.separator+"trans";
	    
	    model = trainer.trainModel(td, tp);
	}

	@Override
	public List<Double> predict(InstanceContainer<PrimalInstance> testdata)
	{
		List<Double> preds = new ArrayList<Double>();
		
		for(PrimalInstance pi: testdata)
	    {
	    	double d = model.classify( transform(pi) );
	    	preds.add(d);
	    }
		
		return preds;
	}

	protected boolean validConfig;
	
	private void setup(InstanceContainer<PrimalInstance> trainingdata, KernelFunction<PrimalInstance> k_fun, SVMTrainingParams params)
	{
		validConfig = true;

		if (k_fun instanceof RBFKernel)
		{
			tp.getKernelParameters().kernel_type = KernelParam.RBF;
			RBFKernel rk = (RBFKernel) k_fun;
			tp.getKernelParameters().rbf_gamma = (Double) rk.getParameter(RBFKernel.GAMMA).getCurrentValue();
		} 
		else if (k_fun instanceof PolyKernel)
		{
			tp.getKernelParameters().kernel_type = KernelParam.POLYNOMIAL;
			PolyKernel pk = (PolyKernel) k_fun;
			tp.getKernelParameters().poly_degree = (Integer) pk.getParameter(PolyKernel.DEGREE).getCurrentValue();
		} 
		else if (k_fun instanceof LinearKernel)
		{
			tp.getKernelParameters().kernel_type = KernelParam.LINEAR;
		} 
		else
		{
			validConfig = false;
		}
		
		tp.getLearningParameters().svm_c = params.getValue(SVMTrainingParams.C);
	}
	
	@Override
	public double getObjectiveValue(int i)
	{
		return 0.0;
	}
	
	private static LabeledFeatureVector transform(PrimalInstance pi)
	{
		int      D    = pi.getNumberOfFeatures();
		
		int[]    dims = new int[D];
		double[] vals = pi.getFeatures();
		
		for(int d=0;d<pi.getNumberOfFeatures();d++)
		{
			dims[d] = d+1;
		}
		
		return new LabeledFeatureVector(pi.getLabel().getDoubleValue(), dims, vals);
	}
	
	private static LabeledFeatureVector transformU(PrimalInstance pi)
	{
		int      D    = pi.getNumberOfFeatures();
		
		int[]    dims = new int[D];
		double[] vals = pi.getFeatures();
		
		for(int d=0;d<pi.getNumberOfFeatures();d++)
		{
			dims[d] = d+1;
		}
		
		return new LabeledFeatureVector(0.0, dims, vals);
	}

	private static LabeledFeatureVector[] transform(InstanceContainer<PrimalInstance> data)
	{
		int N = data.size();
		LabeledFeatureVector[] edata = new LabeledFeatureVector[N];
		for(int i=0;i<data.size();i++)
		{
			edata[i] = transform(data.get(i));
		}
		return edata;
	}

}
