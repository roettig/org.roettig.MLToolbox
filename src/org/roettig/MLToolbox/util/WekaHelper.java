/**
 * 
 */
package org.roettig.MLToolbox.util;


import java.util.Set;

import org.roettig.MLToolbox.base.instance.Instance;
import org.roettig.MLToolbox.base.instance.InstanceContainer;
import org.roettig.MLToolbox.base.instance.PrimalInstance;
import org.roettig.MLToolbox.base.label.FactorLabel;
import org.roettig.MLToolbox.base.label.Label;
import org.roettig.MLToolbox.base.label.NumericLabel;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;

/**
 * @author roettig
 *
 */
public class WekaHelper
{
	public static weka.core.Instance convert(PrimalInstance inst) throws Exception
	{
		FastVector atts = new FastVector();
		
		for(int i=0;i<inst.getNumberOfFeatures();i++)
			atts.addElement(new Attribute(String.format("att%d",i)));

		double[] vals = new double[inst.getNumberOfFeatures()];

		for(int f=0;f<inst.getNumberOfFeatures();f++)
		{
			vals[f] = inst.getFeatures()[f];
		}

		return new weka.core.Instance(1.0, vals);
	}

	public static Instances convert(InstanceContainer<PrimalInstance> samples)
	{
		FastVector atts = new FastVector();

		boolean classification = samples.isFactorLabelled(); 

		if(samples.get(0) instanceof PrimalInstance)
		{

			PrimalInstance pi = (PrimalInstance) samples.get(0);
			
			for(int i=0;i<pi.getNumberOfFeatures();i++)
				atts.addElement(new Attribute(String.format("att%d",i)));


			FastVector attVals = new FastVector();

			if(classification)
			{
				 
				Set<Label> labels = MLHelper.getLabelSet(samples); 

				for(Label l: labels)
				{
					attVals.addElement(l.toString());
				}

				atts.addElement(new Attribute("class", attVals));
			}
			else
			{
				atts.addElement(new Attribute("y"));
			}

			Instances data = new Instances("MyRelation", atts, 0);
			data.setClassIndex(data.numAttributes() - 1);


			for(Instance i: samples)
			{
				pi = (PrimalInstance) i;
				double[] vals = new double[data.numAttributes()];

				for(int f=0;f<pi.getNumberOfFeatures();f++)
				{
					vals[f] = pi.getFeatures()[f];
				}

				if(classification)
				{
					FactorLabel lab = (FactorLabel) pi.getLabel();
					vals[vals.length-1] = attVals.indexOf(lab.toString());
				}
				else
				{
					NumericLabel l      = (NumericLabel) pi.getLabel();
					vals[vals.length-1] = l.getDoubleValue();
				}

				weka.core.Instance wi = new weka.core.Instance(1.0, vals);
				data.add(wi);
			}
			return data;

		}
		return null;
	}
}
