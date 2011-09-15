package org.roettig.MLToolbox.model;


import java.util.ArrayList;
import java.util.List;

import org.roettig.MLToolbox.base.Prediction;
import org.roettig.MLToolbox.base.instance.InstanceContainer;
import org.roettig.MLToolbox.base.instance.PrimalInstance;
import org.roettig.MLToolbox.base.label.Label;
import org.roettig.MLToolbox.base.label.NumericLabel;
import org.roettig.MLToolbox.base.parameter.Parameter;
import org.roettig.MLToolbox.util.WekaHelper;
import org.roettig.MLToolbox.validation.PearsonMeasure;

import weka.classifiers.trees.M5P;
import weka.core.Instances;

public class RegressionTreeM5P extends Model<PrimalInstance> implements ClassificationModel
{

	public static String M = RegressionTreeM5P.class.getCanonicalName()+"_M";
	
	private Parameter<Integer> m = new Parameter<Integer>(M,new Integer[]{5});
	
	private M5P classifier;
	private Instances data;
	
	public RegressionTreeM5P()
	{
		this.registerParameter(M, m);
		qm = new PearsonMeasure();
	}
	
	public void addM(Integer... Ms)
	{
		m = new Parameter<Integer>(M,Ms);
		this.registerParameter(M, m);
	}
	
	@Override
	public void train(InstanceContainer<PrimalInstance> trainingdata)
	{
		classifier = null;
		classifier = new M5P();
		
		classifier.setUnpruned(false);
		
		classifier.setBuildRegressionTree(false);
		classifier.setMinNumInstances(m.getCurrentValue());
		
		this.data = WekaHelper.convert(trainingdata);

		classifier.setMinNumInstances( m.getCurrentValue()  );
		
		try
		{
			classifier.buildClassifier(data);
		} 
		catch (Exception e)
		{
			throw new RuntimeException("could not train Weka M5P model");
		}
	}

	@Override
	public List<Prediction> predict(InstanceContainer<PrimalInstance> testdata)
	{
		List<Prediction> preds = new ArrayList<Prediction>();
		for(PrimalInstance pi: testdata)
		{
			Label yp = predict(pi);
			preds.add(new Prediction(pi, pi.getLabel(), yp) );
		}
		return preds;
	}
	
	private Label predict(PrimalInstance testdata)
	{
		weka.core.Instance inst = WekaHelper.convert(testdata);
		inst.setDataset(data);
		
		double clsLabel;
		try
		{
			clsLabel = classifier.classifyInstance(inst);
		} 
		catch (Exception e)
		{
			throw new RuntimeException("could not predict given instance with M5P model");
		}
		
		return new NumericLabel(clsLabel);
	}

}
