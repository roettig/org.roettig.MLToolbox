package org.roettig.MLToolbox.model;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.roettig.MLToolbox.base.Prediction;
import org.roettig.MLToolbox.base.instance.InstanceContainer;
import org.roettig.MLToolbox.base.instance.PrimalInstance;
import org.roettig.MLToolbox.base.label.Label;
import org.roettig.MLToolbox.base.parameter.Parameter;
import org.roettig.MLToolbox.validation.FMeasure;

import jnisvmlight.KernelParam;
import jnisvmlight.LabeledFeatureVector;
import jnisvmlight.SVMLightModel;
import jnisvmlight.SVMLightInterface;
import jnisvmlight.TrainingParameters;

public class TCSVCModel extends Model<PrimalInstance> implements ClassificationModel
{
	public static String C     = CSVCModel.class.getCanonicalName()+"_C";
	public static String GAMMA = CSVCModel.class.getCanonicalName()+"_gamma";
	
	private Parameter<Double> c = new Parameter<Double>(C,new Double[]{1.0});
	private Parameter<Double> g = new Parameter<Double>(GAMMA,new Double[]{1.0});
	
	
	private SVMLightModel model;
	public Label POS;
	public Label NEG;
	private int verb;
	
	public TCSVCModel()
	{
		super();
		this.registerParameter(C, c);
		this.registerParameter(GAMMA, g);
		qm = new FMeasure();
	}
	
	public void setC(Double... Cs)
	{
		c = new Parameter<Double>(C,Cs);
		this.registerParameter(C, c);
	}
	
	public void setGamma(Double... Gs)
	{
		g = new Parameter<Double>(GAMMA,Gs);
		this.registerParameter(GAMMA, g);
	}
	
	
	
	public void setVerbosity(int verb)
	{
		this.verb = verb;
	}
	
	@Override
	public void train(InstanceContainer<PrimalInstance> trainingdata)
	{
		this.trainingdata = trainingdata;
		
		fetchLabels(trainingdata);
		
		SVMLightInterface trainer = new SVMLightInterface();

		Collection<PrimalInstance> unlab = trainingdata.getUnlabeledData();
		
		int N = trainingdata.size();
		int U = unlab.size();
		
	    
	    LabeledFeatureVector[] traindata = transform(trainingdata);
	    
	    LabeledFeatureVector[] td = new LabeledFeatureVector[N+U];
	    
	    for(int i=0;i<N;i++)
	    {
	    	td[i] = traindata[i];
	    }
	    
	    int i=0;
	    for(PrimalInstance pi: unlab)
	    {
	    	td[N+i] = transformU(pi);
	    	i++;
	    }
	    
	    TrainingParameters tp = new TrainingParameters();

	    tp.getKernelParameters().kernel_type = KernelParam.RBF;
	    tp.getKernelParameters().rbf_gamma = (Double) this.getParameter(GAMMA).getCurrentValue();;
	    tp.getLearningParameters().svm_c   = (Double) this.getParameter(C).getCurrentValue();;
	    
	    // Switch on some debugging output
	    tp.getLearningParameters().verbosity = verb;
	    tp.getLearningParameters().svm_costratio_unlab = 10.0;
	    tp.getLearningParameters().svm_c_factor = 2.0;
	    //tp.getLearningParameters().transduction_posratio = 0.5;
	    
	    tp.getLearningParameters().predfile = "/tmp/trans";
	    
	    model = trainer.trainModel(td, tp);
	}

	@Override
	public List<Prediction> predict(InstanceContainer<PrimalInstance> testdata)
	{
		List<Prediction> preds = new ArrayList<Prediction>();
		
		for(PrimalInstance pi: testdata)
	    {
	    	double d = model.classify( transform(pi) );
	    	Prediction pred = null;
	    	if(d>=0.0)
	    		pred = new Prediction(pi,pi.getLabel(), POS);
	    	else
	    		pred = new Prediction(pi,pi.getLabel(), NEG);
	    	
	    	preds.add(pred);
	    }
		
		return preds;
	}
	
	
	
	private void fetchLabels(InstanceContainer<PrimalInstance> data)
	{
		fetchPosLabel(data);
		fetchNegLabel(data);
	}
	
	private void fetchPosLabel(InstanceContainer<PrimalInstance> data)
	{
		for(PrimalInstance pi: data)
		{
			if(pi.getLabel().getDoubleValue()==1.0)
			{
				POS = pi.getLabel();
				return;
			}
		}
	}

	private void fetchNegLabel(InstanceContainer<PrimalInstance> data)
	{
		for(PrimalInstance pi: data)
		{
			if(pi.getLabel().getDoubleValue()==-1.0)
			{
				NEG = pi.getLabel();
				return;
			}
		}
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
