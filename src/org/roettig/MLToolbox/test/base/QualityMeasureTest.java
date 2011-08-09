package org.roettig.MLToolbox.test.base;

import java.util.ArrayList;
import java.util.List;

import org.roettig.MLToolbox.base.Prediction;
import org.roettig.MLToolbox.base.label.FactorLabel;
import org.roettig.MLToolbox.base.label.Label;
import org.roettig.MLToolbox.validation.QualityMeasure;
import org.roettig.MLToolbox.validation.Sensitivity;

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
}
