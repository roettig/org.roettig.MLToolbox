package org.roettig.MLToolbox.model;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.roettig.MLToolbox.base.Prediction;
import org.roettig.MLToolbox.base.instance.Instance;
import org.roettig.MLToolbox.base.instance.InstanceContainer;
import org.roettig.MLToolbox.base.instance.PrimalInstance;
import org.roettig.MLToolbox.base.label.Label;
import org.roettig.MLToolbox.base.parameter.Parameter;
import org.roettig.MLToolbox.util.WekaHelper;
import org.roettig.MLToolbox.validation.FMeasure;


import weka.classifiers.trees.J48;
import weka.core.Instances;

public class DecisionTreeJ48 extends Model<PrimalInstance> implements ClassificationModel
{
	public static String M = DecisionTreeJ48.class.getCanonicalName()+"_M";
	public static String C = DecisionTreeJ48.class.getCanonicalName()+"_C";
	
	private Parameter<Integer> m = new Parameter<Integer>(M,new Integer[]{5});
	private Parameter<Double>  c = new Parameter<Double>(C,new Double[]{0.5});
	
	private J48   classifier;
	private Instances data;
	
	private Map<String,Label> str2label = new HashMap<String,Label>();
	
	public DecisionTreeJ48()
	{
		this.registerParameter(C, c);
		this.registerParameter(M, m);
		qm = new FMeasure();
	}
	
	public void addM(Integer... Ms)
	{
		m = new Parameter<Integer>(M,Ms);
		this.registerParameter(M, m);
	}
	
	public void addC(Double... Cs)
	{
		c = new Parameter<Double>(C,Cs);
		this.registerParameter(C, c);
	}
	
	@Override
	public void train(InstanceContainer<PrimalInstance> trainingdata)
	{
		classifier = null;
        classifier = new J48();
        
        classifier.setUnpruned(false);
        
        this.data = WekaHelper.convert(trainingdata);
        str2label.clear();
        
        for(Instance i: trainingdata)
        {
                Label lab = i.getLabel();
                str2label.put(lab.toString(), lab);
        }
        try
        {
                classifier.buildClassifier(data);
        } 
        catch (Exception e)
        {
                throw new RuntimeException("could not train Weka J48 model");
        }
	}

	@Override
	public List<Prediction> predict(InstanceContainer<PrimalInstance> testdata)
	{
		List<Prediction> preds = new ArrayList<Prediction>();
		
		weka.core.Instances insts = WekaHelper.convert(testdata);
		
        //inst.setDataset(trainingdata);
		
		for(int i=0;i<insts.numInstances();i++)
		{
			weka.core.Instance inst =  insts.instance(i);
			
			double[] scores;
			try
			{
				scores = classifier.distributionForInstance(inst);
			} 
			catch (Exception e)
			{
				throw new RuntimeException("could not predict given instance with J48 model");
			}
			
			double mx  = -1;
			int    idx =  0;
			int    cls =  0;
			
			// find best scoring class index
			for (double d : scores)
			{
				if (d > mx)
				{
					mx = d;
					cls = idx;
				}
				idx++;
			}
			
			String lab = data.classAttribute().value(cls);
			/*
			idx = 0;
			for (double d : scores)
			{
				pred.addScore(str2label.get(trainingdata.classAttribute().value(idx)),d);
				idx++;
			}
			*/
			Prediction pred = new Prediction(testdata.get(i), testdata.get(i).getLabel(), str2label.get(lab));
			preds.add(pred);
		}
        return preds;
	}

}
