package org.roettig.MLToolbox.test.base;

import java.util.ArrayList;
import java.util.List;

import org.roettig.MLToolbox.base.Prediction;
import org.roettig.MLToolbox.base.label.FactorLabel;
import org.roettig.MLToolbox.base.label.Label;
import org.roettig.MLToolbox.validation.FMeasure;
import org.roettig.MLToolbox.validation.MCC;
import org.roettig.MLToolbox.validation.Precision;
import org.roettig.MLToolbox.validation.QualityMeasure;
import org.roettig.MLToolbox.validation.Sensitivity;
import org.roettig.MLToolbox.validation.Specificity;
import org.roettig.maths.statistics.Statistics;

import junit.framework.TestCase;

public class QualityMeasureTest extends TestCase
{
	public void testSensitivity()
	{
		List<Prediction> preds = new ArrayList<Prediction>();
		FactorLabel pos = new FactorLabel("pos");
		FactorLabel neg = new FactorLabel("neg");
		preds.add(new Prediction(null,pos,neg));
		preds.add(new Prediction(null,pos,pos));
		preds.add(new Prediction(null,pos,pos));
		preds.add(new Prediction(null,pos,pos));
		preds.add(new Prediction(null,pos,pos));
		preds.add(new Prediction(null,pos,neg));
		QualityMeasure sens = new Sensitivity(pos);
		double sq = sens.getQuality(preds);
		// tp=4 fn=2
		assertEquals(0.6666666666666666,sq,1e-3);
		
		preds.clear();
		preds.add(new Prediction(null,neg,pos));
		preds.add(new Prediction(null,pos,pos));
		preds.add(new Prediction(null,pos,pos));
		preds.add(new Prediction(null,pos,pos));
		preds.add(new Prediction(null,pos,pos));
		preds.add(new Prediction(null,neg,pos));
		sq = sens.getQuality(preds);
		// tp=5 fn=0
		assertEquals(1.0,sq,1e-3);
		
		preds.clear();
		preds.add(new Prediction(null,pos,pos));
		preds.add(new Prediction(null,pos,pos));
		preds.add(new Prediction(null,pos,pos));
		preds.add(new Prediction(null,pos,pos));
		preds.add(new Prediction(null,pos,pos));
		preds.add(new Prediction(null,pos,neg));
		sq = sens.getQuality(preds);
		// tp=5 fn=1
		assertEquals(0.83333,sq,1e-3);
	}
	
	public void testSpecificity()
	{
		List<Prediction> preds = new ArrayList<Prediction>();
		
		FactorLabel pos = new FactorLabel("pos");
		FactorLabel neg = new FactorLabel("neg");
		
		preds.add(new Prediction(null,neg,neg));
		preds.add(new Prediction(null,pos,pos));
		preds.add(new Prediction(null,neg,neg));
		preds.add(new Prediction(null,pos,pos));
		preds.add(new Prediction(null,neg,pos));
		preds.add(new Prediction(null,neg,neg));
		
		QualityMeasure spec = new Specificity(pos);
		double sq = spec.getQuality(preds);
		// tn=3 fp=1
		assertEquals(0.75,sq,1e-3);
	}
	
	public void testPrecision()
	{
		List<Prediction> preds = new ArrayList<Prediction>();
		
		FactorLabel pos = new FactorLabel("pos");
		FactorLabel neg = new FactorLabel("neg");
		
		preds.add(new Prediction(null,neg,neg));
		preds.add(new Prediction(null,pos,pos));
		preds.add(new Prediction(null,neg,neg));
		preds.add(new Prediction(null,pos,pos));
		preds.add(new Prediction(null,neg,pos));
		preds.add(new Prediction(null,neg,neg));
		
		QualityMeasure prec = new Precision(pos);
		double sq = prec.getQuality(preds);
		// tp=2 fp=1
		assertEquals(0.6666666666666666,sq,1e-3);
	}
	
	public void testFMeasure()
	{
		List<Prediction> preds = new ArrayList<Prediction>();
		
		FactorLabel pos = new FactorLabel("pos");
		FactorLabel neg = new FactorLabel("neg");
		
		preds.add(new Prediction(null,neg,neg));
		preds.add(new Prediction(null,pos,pos));
		preds.add(new Prediction(null,neg,neg));
		preds.add(new Prediction(null,pos,pos));
		preds.add(new Prediction(null,neg,pos));
		preds.add(new Prediction(null,neg,neg));
		
		QualityMeasure pr = new Precision(pos);
		QualityMeasure re = new Sensitivity(pos);
		double  prv = pr.getQuality(preds);
		double  rev = re.getQuality(preds);
		System.out.println(prv);
		System.out.println(rev);
		QualityMeasure fm = new FMeasure(pos);
		double sq = fm.getQuality(preds);
		// tp=2 fp=1 tn=2 fn=0
		assertEquals(0.8,sq,1e-5);
	}
	
	public void testMCC()
	{
		List<Prediction> preds = new ArrayList<Prediction>();
		FactorLabel pos = new FactorLabel("pos");
		FactorLabel neg = new FactorLabel("neg");
		preds.add(new Prediction(null,pos,pos));
		preds.add(new Prediction(null,pos,pos));
		preds.add(new Prediction(null,pos,pos));
		preds.add(new Prediction(null,neg,pos));
		preds.add(new Prediction(null,neg,pos));
		preds.add(new Prediction(null,neg,neg));
		QualityMeasure mcc = new MCC(pos);
		assertEquals(0.4472135954999579,mcc.getQuality(preds),1e-5);
	}
}
