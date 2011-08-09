package org.roettig.MLToolbox.test.base;

import java.io.IOException;
import java.util.List;
import java.util.Vector;

import org.roettig.MLToolbox.base.impl.DefaultInstanceContainer;
import org.roettig.MLToolbox.base.instance.PrimalInstance;
import org.roettig.MLToolbox.test.data.DataSource;
import org.roettig.MLToolbox.util.InstanceReader;
import org.roettig.MLToolbox.validation.ModelValidation;

import junit.framework.TestCase;


public class ModelValidationTest extends TestCase
{
	public void testGetFold() throws IOException
	{

		DefaultInstanceContainer<PrimalInstance> samples = InstanceReader.read(DataSource.class.getResourceAsStream("test.dat"), 3, true);
		int idx = 1;
		for(PrimalInstance i: samples)
		{
			i.addProperty("idx", idx);
			idx++;
		}
		DefaultInstanceContainer<PrimalInstance> train = new DefaultInstanceContainer<PrimalInstance>();
		DefaultInstanceContainer<PrimalInstance> test  = new DefaultInstanceContainer<PrimalInstance>();


		ModelValidation.getFold(1, 2, samples, train, test);
		idx = 1;
		for(PrimalInstance i: train)
		{
			if(idx==1)
			{
				assertEquals(2,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			if(idx==2)
			{
				assertEquals(4,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==3)
			{
				assertEquals(6,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			if(idx==4)
			{
				assertEquals(8,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			idx++;
		}
		idx=1;
		for(PrimalInstance i: test)
		{
			if(idx==1)
			{
				assertEquals(1,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==2)
			{
				assertEquals(3,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==3)
			{
				assertEquals(5,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			if(idx==4)
			{
				assertEquals(7,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			idx++;
		}

		train.clear();
		test.clear();
		ModelValidation.getFold(2, 2, samples, train, test);
		idx = 1;
		for(PrimalInstance i: test)
		{

			if(idx==1)
			{
				assertEquals(2,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			if(idx==2)
			{
				assertEquals(4,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==3)
			{
				assertEquals(6,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			if(idx==4)
			{
				assertEquals(8,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			idx++;
		}
		idx=1;
		for(PrimalInstance i: train)
		{

			if(idx==1)
			{
				assertEquals(1,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==2)
			{
				assertEquals(3,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==3)
			{
				assertEquals(5,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			if(idx==4)
			{
				assertEquals(7,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			idx++;
		}

		train.clear();
		test.clear();
		ModelValidation.getFold(1, 3, samples, train, test);
		idx = 1;
		for(PrimalInstance i: train)
		{
			if(idx==1)
			{
				assertEquals(2,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			if(idx==2)
			{
				assertEquals(3,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==3)
			{
				assertEquals(5,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			if(idx==4)
			{
				assertEquals(6,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			if(idx==5)
			{
				assertEquals(8,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}	    
			idx++;
		}
		idx=1;
		for(PrimalInstance i: test)
		{
			if(idx==1)
			{
				assertEquals(1,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==2)
			{
				assertEquals(4,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==3)
			{
				assertEquals(7,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			idx++;
		}

		train.clear();
		test.clear();
		ModelValidation.getFold(2, 3, samples, train, test);
		idx = 1;
		for(PrimalInstance i: train)
		{
			if(idx==1)
			{
				assertEquals(1,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==2)
			{
				assertEquals(3,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==3)
			{
				assertEquals(4,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==4)
			{
				assertEquals(6,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			if(idx==5)
			{
				assertEquals(7,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}	    
			idx++;
		}
		idx=1;
		for(PrimalInstance i: test)
		{
			if(idx==1)
			{
				assertEquals(2,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			if(idx==2)
			{
				assertEquals(5,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			if(idx==3)
			{
				assertEquals(8,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			idx++;
		}

		train.clear();
		test.clear();
		ModelValidation.getFold(3, 3, samples, train, test);
		idx = 1;
		for(PrimalInstance i: train)
		{
			if(idx==1)
			{
				assertEquals(1,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==2)
			{
				assertEquals(2,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			if(idx==3)
			{
				assertEquals(4,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==4)
			{
				assertEquals(5,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			if(idx==5)
			{
				assertEquals(7,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==6)
			{
				assertEquals(8,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			idx++;
		}

		idx=1;
		for(PrimalInstance i: test)
		{

			if(idx==1)
			{
				assertEquals(3,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==2)
			{
				assertEquals(6,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			idx++;
		}
	}
	
	public void testGetStratifiedFold() throws IOException
	{
		DefaultInstanceContainer<PrimalInstance> samples = InstanceReader.read(DataSource.class.getResourceAsStream("test.dat"), 3, true);
		int idx = 1;
		for(PrimalInstance i: samples)
		{
			i.addProperty("idx", idx);
			idx++;
		}
		
		DefaultInstanceContainer<PrimalInstance> train = new DefaultInstanceContainer<PrimalInstance>();
		DefaultInstanceContainer<PrimalInstance> test  = new DefaultInstanceContainer<PrimalInstance>();

		ModelValidation.getStratifiedFold(1, 2, samples, train, test);
		idx = 1;
		for(PrimalInstance i: train)
		{
			if(idx==1)
			{
				assertEquals(3,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==2)
			{
				assertEquals(7,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==3)
			{
				assertEquals(5,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			if(idx==4)
			{
				assertEquals(8,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			idx++;
		}

		idx=1;
		for(PrimalInstance i: test)
		{
			if(idx==1)
			{
				assertEquals(1,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==2)
			{
				assertEquals(4,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==3)
			{
				assertEquals(2,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			if(idx==4)
			{
				assertEquals(6,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			idx++;
		}

		train.clear();
		test.clear();
		ModelValidation.getStratifiedFold(2, 2, samples, train, test);
		idx = 1;
		for(PrimalInstance i: test)
		{
			if(idx==1)
			{
				assertEquals(3,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==2)
			{
				assertEquals(7,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==3)
			{
				assertEquals(5,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			if(idx==4)
			{
				assertEquals(8,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			idx++;
		}
		idx = 1;
		for(PrimalInstance i: train)
		{
			if(idx==1)
			{
				assertEquals(1,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==2)
			{
				assertEquals(4,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==3)
			{
				assertEquals(2,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			if(idx==4)
			{
				assertEquals(6,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			idx++;
		}

		train.clear();
		test.clear();
		ModelValidation.getStratifiedFold(1, 3, samples, train, test);
		idx = 1;
		for(PrimalInstance i: train)
		{
			if(idx==1)
			{
				assertEquals(3,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==2)
			{
				assertEquals(4,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==3)
			{
				assertEquals(5,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			if(idx==4)
			{
				assertEquals(6,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			idx++;
		}

		idx = 1;
		for(PrimalInstance i: test)
		{
			if(idx==1)
			{
				assertEquals(1,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==2)
			{
				assertEquals(7,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==3)
			{
				assertEquals(2,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			if(idx==4)
			{
				assertEquals(8,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			idx++;
		}


		train.clear();
		test.clear();
		ModelValidation.getStratifiedFold(2, 3, samples, train, test);
		idx = 1;
		for(PrimalInstance i: train)
		{
			if(idx==1)
			{
				assertEquals(1,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==2)
			{
				assertEquals(4,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==3)
			{
				assertEquals(7,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==4)
			{
				assertEquals(2,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			if(idx==5)
			{
				assertEquals(6,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			if(idx==6)
			{
				assertEquals(8,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}	    
			idx++;
		}

		idx = 1;
		for(PrimalInstance i: test)
		{
			if(idx==1)
			{
				assertEquals(3,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==2)
			{
				assertEquals(5,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			idx++;
		}

		train.clear();
		test.clear();
		ModelValidation.getStratifiedFold(3, 3, samples, train, test);
		idx = 1;
		for(PrimalInstance i: train)
		{
			if(idx==1)
			{
				assertEquals(1,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==2)
			{
				assertEquals(3,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==3)
			{
				assertEquals(7,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==4)
			{
				assertEquals(2,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			if(idx==5)
			{
				assertEquals(5,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			if(idx==6)
			{
				assertEquals(8,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}	    
			idx++;
		}

		idx = 1;
		for(PrimalInstance i: test)
		{
			if(idx==1)
			{
				assertEquals(4,i.getProperty("idx"));
				assertEquals("class1",i.getLabel().toString());
			}
			if(idx==2)
			{
				assertEquals(6,i.getProperty("idx"));
				assertEquals("class2",i.getLabel().toString());
			}
			idx++;
		}
	}
	
	public void testGetStratifiedSplit() throws IOException
	{
		DefaultInstanceContainer<PrimalInstance> samples = InstanceReader.read(DataSource.class.getResourceAsStream("test.dat"), 3, true);
		
		int idx = 1;
		for(PrimalInstance i: samples)
		{
			i.addProperty("idx", idx);
			idx++;
		}

		DefaultInstanceContainer<PrimalInstance> train = new DefaultInstanceContainer<PrimalInstance>();
		DefaultInstanceContainer<PrimalInstance> test  = new DefaultInstanceContainer<PrimalInstance>();

		ModelValidation.getStratifiedSplit(0.5, samples, train, test);
		idx = 1;
		for(PrimalInstance i: train)
		{
			int inst_idx = (Integer) i.getProperty("idx");
			if(idx==1)
			{
				assertEquals(inst_idx,1);
				assertEquals(i.getLabel().toString(),"class1");
			}
			if(idx==2)
			{
				assertEquals(inst_idx,3);
				assertEquals(i.getLabel().toString(),"class1");
			}
			if(idx==3)
			{
				assertEquals(inst_idx,2);
				assertEquals(i.getLabel().toString(),"class2");
			}
			if(idx==4)
			{
				assertEquals(inst_idx,5);
				assertEquals(i.getLabel().toString(),"class2");
			}
			idx++;
		}

		idx = 1;
		for(PrimalInstance i: test)
		{

			int inst_idx = (Integer) i.getProperty("idx");
			if(idx==1)
			{
				assertEquals(inst_idx,4);
				assertEquals(i.getLabel().toString(),"class1");
			}
			if(idx==2)
			{
				assertEquals(inst_idx,7);
				assertEquals(i.getLabel().toString(),"class1");
			}
			if(idx==3)
			{
				assertEquals(inst_idx,6);
				assertEquals(i.getLabel().toString(),"class2");
			}
			if(idx==4)
			{
				assertEquals(inst_idx,8);
				assertEquals(i.getLabel().toString(),"class2");
			}
			idx++;
		}
	}

	public <T> boolean compareArray(T[] a, T[] b)
	{
		if(a.length!=b.length)
			return false;
		for(int i=0;i<a.length;i++)
		{
			if(!a[i].equals(b[i]))
				return false;
		}
		return true;
	}
	
	public void testGetGridPoints()
	{

		int size[] = {3,1,2};
		Vector<Integer[]> expected = new Vector<Integer[]>();
		expected.add( new Integer[] {0,0,0} );
		expected.add( new Integer[] {1,0,0} );
		expected.add( new Integer[] {2,0,0} );
		expected.add( new Integer[] {0,0,1} );
		expected.add( new Integer[] {1,0,1} );
		expected.add( new Integer[] {2,0,1} );
		Vector<Integer[]> idx =  ModelValidation.getGridPoints( size );
		int i=0;
		for(Integer[] inta : idx)
		{
			assertTrue(compareArray(inta,expected.get(i)));
			i++;
		}
	}
}
