package org.roettig.MLToolbox.test.base;

import org.roettig.MLToolbox.base.instance.PrimalInstance;
import org.roettig.MLToolbox.base.label.FactorLabel;
import org.roettig.MLToolbox.kernels.LinearKernel;
import org.roettig.MLToolbox.kernels.RBFKernel;

import junit.framework.TestCase;


public class KernelTest extends TestCase
{
	public void testRBF() throws Exception
	{
		RBFKernel rbf = new RBFKernel();
		rbf.setGamma(0.01);
		PrimalInstance pi1 = new PrimalInstance(new FactorLabel("class1"),new double[]{0.1});
		PrimalInstance pi2 = new PrimalInstance(new FactorLabel("class1"),new double[]{0.3});
		assertEquals(rbf.compute(pi1, pi2),0.9996000799893344,1e-6);
		assertEquals(rbf.computeN(pi1, pi2),0.9996000799893344,1e-6);
		
		PrimalInstance pi3 = new PrimalInstance(new FactorLabel("class1"),new double[]{-0.3});
		assertEquals(rbf.compute(pi1, pi3),0.9984012793176064,1e-6);
		assertEquals(rbf.computeN(pi1, pi3),.9984012793176064,1e-6);
		
		PrimalInstance pi4 = new PrimalInstance(new FactorLabel("class1"),new double[]{0.1,0.2});
		PrimalInstance pi5 = new PrimalInstance(new FactorLabel("class2"),new double[]{0.3,0.1});
		
		LinearKernel lk = new LinearKernel();
		
		assertEquals(lk.compute(pi4, pi5),0.05,1e-6);
		assertEquals(lk.computeN(pi4, pi5),0.7071067811865475,1e-6);
	}
}
