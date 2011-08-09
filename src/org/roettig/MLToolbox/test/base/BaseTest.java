package org.roettig.MLToolbox.test.base;

import junit.framework.TestCase;

import org.roettig.MLToolbox.base.instance.PrimalInstance;
import org.roettig.MLToolbox.base.label.FactorLabel;
import org.roettig.MLToolbox.base.label.Label;
import org.roettig.MLToolbox.base.label.NumericLabel;


public class BaseTest extends TestCase
{
	public void testLabel()
	{
		FactorLabel lab1 = new FactorLabel("class1");
		FactorLabel lab2 = new FactorLabel("class2");
		FactorLabel lab3 = new FactorLabel("class1");
		
		assertTrue(lab1.equals(lab3));
		assertTrue(!lab1.equals(lab2));
		
		Label lab4 = new NumericLabel(22.0);
		Label lab5 = new NumericLabel(22.0);
		Label lab6 = new NumericLabel(2.1);

		assertTrue(lab4.equals(lab5));
		assertTrue(!lab4.equals(lab6));
		
		System.out.println(lab1.getDoubleValue());
		System.out.println(lab2.getDoubleValue());
	}
	
	public void testInstance()
	{
		Label lab1 = new FactorLabel("class1");
		Label lab2 = new FactorLabel("class2");
		
		double[] fts1 = new double[]{1.0,2.0};
		double[] fts2 = new double[]{-1.0,-2.0};
		
		
		PrimalInstance pi1 = new PrimalInstance(lab1,fts1);
		PrimalInstance pi2 = new PrimalInstance(lab2,fts2);
		
		assertEquals(pi1.getId(),0);
		assertEquals(pi2.getId(),1);
		
		assertEquals(pi1.getFeatures()[0],1.0);
		assertEquals(pi1.getFeatures()[1],2.0);
		
		assertEquals(pi2.getFeatures()[0],-1.0);
		assertEquals(pi2.getFeatures()[1],-2.0);
		
	}
}
