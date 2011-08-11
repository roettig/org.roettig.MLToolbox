package org.roettig.MLToolbox.util;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;

import org.roettig.MLToolbox.base.instance.Instance;
import org.roettig.MLToolbox.base.instance.InstanceContainer;
import org.roettig.MLToolbox.base.instance.PrimalInstance;
import org.roettig.MLToolbox.base.label.Label;

/**
 * The MLHelper offers several usefull helper methods for the MLToolbox.
 * 
 * @author roettig
 *
 */
public class MLHelper
{
	/**
	 * returns the set of all labels within an instance container.
	 * 
	 * @param data
	 * @return set of all labels
	 */
	public static <T extends Instance> Set<Label> getLabelSet(InstanceContainer<T> data)
	{
		Set<Label> ret = new HashSet<Label>();
		for(T t : data)
		{
			ret.add(t.getLabel());
		}
		return ret;
	}
	
	/**
	 * exports the data in a instance container into a LIBSVM formatted file.
	 * 
	 * @param data
	 * @param filename
	 * @throws Exception
	 */
	public static void exportLIBSVM(InstanceContainer<PrimalInstance> data, String filename) throws Exception
	{
		PrintWriter out = new PrintWriter(new File(filename) );
		for(PrimalInstance pi: data)
		{
			out.print(String.format(Locale.ENGLISH,"%.4f ",pi.getLabel().getDoubleValue()));
			for(int j=0;j<pi.getNumberOfFeatures();j++)
				out.print(String.format(Locale.ENGLISH,"%d:%.4f ",j+1,pi.getFeatures()[j]));
			out.println();
		}
		out.close();
	}
	
	/**
	 * returns a list of indices corresponding to a random permutation of a list of indices from 0 to n-1.
	 * @param n
	 * @param rng
	 * 
	 * @return list of permuted indices
	 */
	public static List<Integer> randPerm(int n, Random rng)
	{
		List<Integer> ret = indexList(n);
		Collections.shuffle(ret,rng);
		return ret;
	}

	/**
	 * returns a list of indices from 0 to n-1.
	 * 
	 * @param n
	 * 
	 * @return list of indices
	 */
	public static List<Integer> indexList(int n)
	{
		List<Integer> ret = new ArrayList<Integer>();
		for(int i=0;i<n;i++)
		{
			ret.add(i);   
		}
		return ret;
	}
	
	/**
	 * computes the mean distance of primal instances stored in a container.
	 * 
	 * @param data
	 * @return mean distance
	 */
	public static double meanDistance(InstanceContainer<PrimalInstance> data)
	{
		double ret = 0.0;
		int    N   = 0;
		for(int i=0;i<data.size();i++)
		{
			for(int j=(i+1);j<data.size();j++)
			{
				ret+= PrimalInstance.squaredDistance(data.get(i), data.get(j));
				N++;
			}
		}
		return ret/N;
	}
	
	/**
	 * creates a list of exponentiated values (factor*10^k) with exponents <i>k</i> ranging from start to end with the given stride.
	 * 
	 * @param start first exponent value
	 * @param end last exponent value
	 * @param stride stride
	 * @param factor
	 * @return array of doubles
	 */
	public static Double[] makeExpSequence(int start, int end, int stride, double factor)
	{
		List<Double> elems = new ArrayList<Double>();
		int s = start;
		while(s<end)
		{
			elems.add(Math.pow(10,s)*factor);
			s+=stride;
		}
		int N = elems.size();
		Double[] ret = new Double[N];
		for(int i=0;i<N;i++)
		{
			ret[i] = elems.get(i);
		}
		return ret;
	}
}
