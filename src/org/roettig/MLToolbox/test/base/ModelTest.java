package org.roettig.MLToolbox.test.base;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import junit.framework.TestCase;

import org.roettig.MLToolbox.base.Prediction;
import org.roettig.MLToolbox.base.impl.DefaultInstanceContainer;
import org.roettig.MLToolbox.base.instance.InstanceContainer;
import org.roettig.MLToolbox.base.instance.PrimalInstance;
import org.roettig.MLToolbox.base.label.FactorLabel;
import org.roettig.MLToolbox.kernels.LinearKernel;
import org.roettig.MLToolbox.kernels.RBFKernel;
import org.roettig.MLToolbox.model.CSVCModel;
import org.roettig.MLToolbox.model.DecisionTreeJ48;
import org.roettig.MLToolbox.model.Model;
import org.roettig.MLToolbox.model.NuSVCModel;
import org.roettig.MLToolbox.model.NuSVRModel;
import org.roettig.MLToolbox.model.OneClassSVM;
import org.roettig.MLToolbox.model.RegressionTreeM5P;
import org.roettig.MLToolbox.model.TCSVCModel;
import org.roettig.MLToolbox.model.kNN;
import org.roettig.MLToolbox.test.data.DataSource;
import org.roettig.MLToolbox.util.InstanceReader;
import org.roettig.MLToolbox.validation.ModelValidation;


public class ModelTest extends TestCase
{
	
	public void testLIBSVM1() throws Exception
	{
		RBFKernel rbf = new RBFKernel();
		rbf.setGamma(0.01);
		rbf.doNormalize(false);
		
		DefaultInstanceContainer<PrimalInstance>  data = InstanceReader.read(DataSource.class.getResourceAsStream("iris.dat"), 5, true);

		CSVCModel<PrimalInstance>                 m  = new CSVCModel<PrimalInstance>(rbf);
		
		m.setC(1.0);

		DefaultInstanceContainer<PrimalInstance>  train = new DefaultInstanceContainer<PrimalInstance>();
		DefaultInstanceContainer<PrimalInstance>  test  = new DefaultInstanceContainer<PrimalInstance>();
		
		ModelValidation.getStratifiedSplit(0.5, data, train, test);
		
		m.train(train);
		List<Prediction> preds = m.predict(test);
		assertEquals(0.9167613421192152,m.getQuality(preds),1e-4);
		double objs[] = {-13.846634,-7.607551,-36.366686};
		for(int i=0;i<3;i++)
		{
			assertEquals(objs[i],m.getObjectiveValue(i),1e-4);
		}
	}
	
	public void testLIBSVM2() throws Exception
	{
		LinearKernel lk = new LinearKernel();
		lk.doNormalize(false);
		
		DefaultInstanceContainer<PrimalInstance>  data = InstanceReader.read(DataSource.class.getResourceAsStream("iris.dat"), 5, true);

		CSVCModel<PrimalInstance>                 m  = new CSVCModel<PrimalInstance>(lk);
		
		m.setC(1.0);

		DefaultInstanceContainer<PrimalInstance>  train = new DefaultInstanceContainer<PrimalInstance>();
		DefaultInstanceContainer<PrimalInstance>  test  = new DefaultInstanceContainer<PrimalInstance>();
		
		ModelValidation.getStratifiedSplit(0.5, data, train, test);
		
		m.train(train);
		List<Prediction> preds = m.predict(test);
		assertEquals(0.9475675798155576,m.getQuality(preds),1e-4);
		double objs[] = {-0.559475,-0.203684,-8.836843};
		for(int i=0;i<3;i++)
		{
			assertEquals(objs[i],m.getObjectiveValue(i),1e-4);
		}
	}
	
	public void testLIBSVM3() throws Exception
	{
		RBFKernel rbf = new RBFKernel();
		rbf.setGamma(1.0);
		
		DefaultInstanceContainer<PrimalInstance> samples = InstanceReader.read(DataSource.class.getResourceAsStream("sin.dat"), 2, false);
		
		NuSVRModel<PrimalInstance>     m  = new NuSVRModel<PrimalInstance>(rbf);
		m.setC(10.0);
		m.setNU(0.2);
		
		DefaultInstanceContainer<PrimalInstance>  train = new DefaultInstanceContainer<PrimalInstance>();
		DefaultInstanceContainer<PrimalInstance>  test  = new DefaultInstanceContainer<PrimalInstance>();
		
		samples.shuffle(new Random(2204));
		
		ModelValidation.getSplit(0.8, samples, train, test);
		
		
		m.train(train);
		List<Prediction> preds = m.predict(test);

		assertEquals(0.9913,m.getQuality(preds),1e-2);
		
		assertEquals(-661.761801,m.getObjectiveValue(0),1e-2);
	}
	
	public void testLIBSVM4() throws Exception
	{
		RBFKernel rbf = new RBFKernel();
		rbf.setGamma(1.0);
		
		DefaultInstanceContainer<PrimalInstance>  data = InstanceReader.read(DataSource.class.getResourceAsStream("iris.dat"), 5, true);

		OneClassSVM<PrimalInstance>                 m  = new OneClassSVM<PrimalInstance>(rbf);
		
		m.setNu(0.05);

		DefaultInstanceContainer<PrimalInstance>  train = new DefaultInstanceContainer<PrimalInstance>();
		DefaultInstanceContainer<PrimalInstance>  test  = new DefaultInstanceContainer<PrimalInstance>();
		
		ModelValidation.getStratifiedSplit(0.80, data, train, test);
		
		m.train(train);
		assertEquals(1.978206,m.getObjectiveValue(0),1e-5);
		
		List<Prediction> preds = m.predict(test);
		double sens = m.getQuality(preds);

		assertEquals(0.8666666666666667,sens,1e-5);
	}
	
	public void testLIBSVM5() throws Exception
	{
		LinearKernel lk = new LinearKernel();
		lk.doNormalize(false);
		
		DefaultInstanceContainer<PrimalInstance>  data = InstanceReader.read(DataSource.class.getResourceAsStream("iris.dat"), 5, true);

		NuSVCModel<PrimalInstance>                 m  = new NuSVCModel<PrimalInstance>(lk);
		
		m.setNu(0.2);

		DefaultInstanceContainer<PrimalInstance>  train = new DefaultInstanceContainer<PrimalInstance>();
		DefaultInstanceContainer<PrimalInstance>  test  = new DefaultInstanceContainer<PrimalInstance>();
		
		ModelValidation.getStratifiedSplit(0.5, data, train, test);
				
		m.train(train);
		
		List<Prediction> preds = m.predict(test);
		assertEquals(0.9475675798155576,m.getQuality(preds),1e-4);
		
		double objs[] = {0.275968,0.126828,4.312706};
		for(int i=0;i<3;i++)
		{
			assertEquals(objs[i],m.getObjectiveValue(i),1e-4);
		}
		
	}
	
	public void testCSVC() throws Exception
	{
		LinearKernel lK = new LinearKernel();
		
		DefaultInstanceContainer<PrimalInstance> samples = InstanceReader.read(DataSource.class.getResourceAsStream("iris.dat"), 5, true);
		
		Model<PrimalInstance>     m  = new CSVCModel<PrimalInstance>(lK);
		
		double qual = ModelValidation.CV(5, samples, m);
		assertEquals(0.9621380846325167,qual,1e-6);
		System.out.println(qual);
	}
		
	public void testNuSVR() throws Exception
	{
		RBFKernel rbf = new RBFKernel();
		rbf.setGamma(1.0);
		
		DefaultInstanceContainer<PrimalInstance> samples = InstanceReader.read(DataSource.class.getResourceAsStream("sin.dat"), 2, false);
		
		NuSVRModel<PrimalInstance>     m  = new NuSVRModel<PrimalInstance>(rbf);
		m.setC(10.0);
		m.setNU(0.2);
		
		List<Prediction> preds = new ArrayList<Prediction>();
		double qual = ModelValidation.CV(5, samples, m, preds);
		
		
		assertEquals(0.9862527328922244,qual,1e-6);
	}
	
	public void testTCSVC1() throws Exception
	{
		RBFKernel k = new RBFKernel();
		k.setGamma(0.001);
		
		TCSVCModel m = new TCSVCModel( k );
		m.setC(100.0);
		
		DefaultInstanceContainer<PrimalInstance>  data = InstanceReader.read(DataSource.class.getResourceAsStream("iris.dat"), 5, true);
		
		InstanceContainer<PrimalInstance> samples = ModelValidation.binaryDivision(data, data.get(51).getLabel(), TCSVCModel.POS, TCSVCModel.NEG);
		
		DefaultInstanceContainer<PrimalInstance>  train = new DefaultInstanceContainer<PrimalInstance>();
		DefaultInstanceContainer<PrimalInstance>  test  = new DefaultInstanceContainer<PrimalInstance>();
		ModelValidation.getStratifiedSplit(0.8, samples, train, test);
		
		m.train(train);
		List<Prediction> preds = m.predict(test);
		
		double q1 = m.getQualityMeasure().getQuality(preds);
		
		FactorLabel pos = new FactorLabel("pos");
		FactorLabel neg = new FactorLabel("neg");
		
		samples = ModelValidation.binaryDivision(data, data.get(51).getLabel(), pos, neg);
		
		train = new DefaultInstanceContainer<PrimalInstance>();
		test  = new DefaultInstanceContainer<PrimalInstance>();
		ModelValidation.getStratifiedSplit(0.8, samples, train, test);
		
		RBFKernel k2 = new RBFKernel();
		k2.setGamma(0.001);
		CSVCModel<PrimalInstance> m2 = new CSVCModel<PrimalInstance>(k2);
		m2.setC(100.0);
		
		m2.train(train);
		preds = m2.predict(test);
		
		double q2 = m2.getQualityMeasure().getQuality(preds);
		
		assertEquals(q1,q2,0.02);
	}
	
	public void testDTreeJ48() throws IOException
	{
		DefaultInstanceContainer<PrimalInstance>  data = InstanceReader.read(DataSource.class.getResourceAsStream("iris.dat"), 5, true);

		DefaultInstanceContainer<PrimalInstance>  train = new DefaultInstanceContainer<PrimalInstance>();
		DefaultInstanceContainer<PrimalInstance>  test  = new DefaultInstanceContainer<PrimalInstance>();
		
		ModelValidation.getStratifiedSplit(0.8, data, train, test);
		
		DecisionTreeJ48 m = new DecisionTreeJ48();
		m.addC(0.4);
		m.addM(5);
		
		m.train(train);
		
		List<Prediction> preds = m.predict(test);
		assertEquals(0.9681794470526864,m.getQuality(preds),1e-5);
	}
	
	public void testkNN() throws IOException
	{
		DefaultInstanceContainer<PrimalInstance>  data = InstanceReader.read(DataSource.class.getResourceAsStream("iris.dat"), 5, true);

		DefaultInstanceContainer<PrimalInstance>  train = new DefaultInstanceContainer<PrimalInstance>();
		DefaultInstanceContainer<PrimalInstance>  test  = new DefaultInstanceContainer<PrimalInstance>();
		
		ModelValidation.getStratifiedSplit(0.5, data, train, test);
		
		kNN<PrimalInstance> m = new kNN<PrimalInstance>(new LinearKernel() ) ;
		m.addK(3);
		m.train(train);
		
		List<Prediction> preds = m.predict(test);
		assertEquals(0.920815165434747,m.getQuality(preds),1e-5);

	}
	
	public void testM5P() throws IOException
	{
		DefaultInstanceContainer<PrimalInstance>  data = InstanceReader.read(DataSource.class.getResourceAsStream("sin.dat"), 2, false);

		DefaultInstanceContainer<PrimalInstance>  train = new DefaultInstanceContainer<PrimalInstance>();
		DefaultInstanceContainer<PrimalInstance>  test  = new DefaultInstanceContainer<PrimalInstance>();
		
		ModelValidation.getSplit(0.8, data, train, test);
		
		RegressionTreeM5P m = new RegressionTreeM5P();
		m.addM(5);
		
		m.train(train);
		
		List<Prediction> preds = m.predict(test);
		assertEquals(0.829477167998876,m.getQuality(preds),1e-5);
	}
}
