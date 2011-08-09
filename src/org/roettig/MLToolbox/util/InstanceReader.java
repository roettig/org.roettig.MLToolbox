package org.roettig.MLToolbox.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
import java.util.Vector;

import org.roettig.MLToolbox.base.impl.DefaultInstanceContainer;
import org.roettig.MLToolbox.base.instance.FilteredDataView;
import org.roettig.MLToolbox.base.instance.InstanceContainer;
import org.roettig.MLToolbox.base.instance.InstanceFilter;
import org.roettig.MLToolbox.base.instance.PrimalInstance;
import org.roettig.MLToolbox.base.label.FactorLabel;
import org.roettig.MLToolbox.base.label.Label;
import org.roettig.MLToolbox.base.label.LabelFactory;
import org.roettig.MLToolbox.base.label.NumericLabel;

public class InstanceReader
{
	
	public static void writeExt(InstanceContainer<PrimalInstance> data, String filename) throws IOException
	{
		FileWriter out = new FileWriter(new File(filename));
		
		for(PrimalInstance t: data)
		{
			out.write(String.format("\"%s\" ",t.getLabel().toString()));
			double[] fts = t.getFeatures();
			for(int i=0;i<fts.length;i++)
			{
				out.write(String.format(Locale.ENGLISH,"%d:%e ",(i+1),fts[i]));
			}
			if(t.getPropertyNames().size()>0)
			{
				out.write(" # ");
				for(String name: t.getPropertyNames())
				{
					out.write(String.format("\"%s\":\"%s\" ",name,t.getProperty(name).toString()));
				}
			}
			out.write("\n");
		}
		out.close();
	}
	
	public static DefaultInstanceContainer<PrimalInstance> readExt(InputStream input, int labelColumn, boolean factor) throws Exception
	{
		labelColumn--;
		
		DefaultInstanceContainer<PrimalInstance> ret = new DefaultInstanceContainer<PrimalInstance>();
		
		//Vector< Vector<Double> > data = new Vector< Vector<Double> >();
		
		BufferedReader bin = new BufferedReader(new InputStreamReader(input));
		
		Map<String,Double> classname_to_double = new HashMap<String,Double>();
		
		
		List<Map<Integer,Double>> data  = new ArrayList<Map<Integer,Double>>();
		List<Map<String,Object>>  properties = new ArrayList<Map<String,Object>>();
		Set<String> propkeys = new HashSet<String>();
		
		List<Label> labels = new ArrayList<Label>();
		
		double cnum = 1.0;
		int    max_ft_idx = -1;
		String line=null;
		while((line=bin.readLine())!=null)
		{
			line = line.trim();
			String toks[] = line.split("#");
			String fts   = toks[0].trim();
			String props = null;
			if(toks.length>1)
				props = toks[1].trim();	
			
			//String f_toks[] = fts.split("\\s+");
			String f_toks[] = fts.split("\\t");
			String classlabel = f_toks[0];
			
			if(classlabel.startsWith("\"")&&classlabel.endsWith("\""))
			{
				classlabel = classlabel.substring(1,classlabel.length()-1);
			}
			else
			{
				
			}
			if(!classname_to_double.containsKey(classlabel))
				classname_to_double.put(classlabel, cnum++);
			
			Label label;
			if(factor)
			{
				label = new FactorLabel(classlabel);
			}
			else
			{
				double val = Double.parseDouble(classlabel);
				label = new NumericLabel(val);
			}
			labels.add(label);
			
			// read features
			Map<Integer,Double> ftsm = new HashMap<Integer,Double>();
			for(int i=1;i<f_toks.length;i++)
			{
				String[] ft_toks = f_toks[i].split(":");
				int    idx;
				try
				{
				 idx = Integer.parseInt(ft_toks[0]);
				}
				catch(Exception e)
				{
					throw new Exception("invalid index found",e);
				}
				if(idx==0)
					throw new Exception("invalid index found");
				if(idx>max_ft_idx)
					max_ft_idx = idx;
				
				double val;
				try
				{
					val = Double.parseDouble(ft_toks[1]);
				}
				catch(Exception e)
				{
					throw new Exception("invalid numeric value found",e);
				}
				ftsm.put(idx, val);
			}
			data.add(ftsm);
			
			if(props==null)
			{
				properties.add(null);
				continue;
			}
			
			String p_toks[] = props.split("\\s+");
			
			Map<String,Object> prop_ = new HashMap<String,Object>();
			
			for(int i=0;i<p_toks.length;i++)
			{
				String[] pp_toks = p_toks[i].split(":");
				String key = pp_toks[0];
				String val = pp_toks[1];
				
				if(key.startsWith("\"")&&key.endsWith("\""))
				{
					key = key.substring(1,key.length()-1);
				}
				
				if(val.startsWith("\"")&&val.endsWith("\""))
				{
					val = val.substring(1,val.length()-1);
					prop_.put(key,val);
					propkeys.add(key);
				}
				else
				{
					
					Double dval=0.0;
					boolean convert_ok = true;
					try
					{
					 dval = Double.parseDouble(val);
					}
					catch(Exception e)
					{
						convert_ok = false;
					}
					if(convert_ok)
						prop_.put(key,dval);
					else
						prop_.put(key,val);
					propkeys.add(key);
				}
			}
			properties.add(prop_);
		}
		
		int i=0;
		for(Map<Integer,Double> row : data)
		{

			double fts[] = new double[max_ft_idx];

			Label label = labels.get(i);
			for(Integer idx: row.keySet())
			{
				fts[idx-1] = row.get(idx);
			}
			
			PrimalInstance pi = new PrimalInstance(label,fts);


			Map<String,Object> pair = properties.get(i);



			for(String key: propkeys)
			{
				if(pair.containsKey(key))
					pi.addProperty(key,pair.get(key));
				else
					pi.addProperty(key,null);
			}				

			
			ret.add(pi);			
			pi.addProperty("id", i+1);			
			i++;
		}
		
		return ret;
	}
	
	public static DefaultInstanceContainer<PrimalInstance> read(InputStream input, int labelColumn, boolean factor) throws IOException
	{
		labelColumn--;
		
		DefaultInstanceContainer<PrimalInstance> ret = new DefaultInstanceContainer<PrimalInstance>();
		
		Vector< Vector<Double> > data = new Vector< Vector<Double> >(); 
		Scanner lineScanner = new Scanner( input ); 
		lineScanner.useDelimiter(System.getProperty("line.separator")); 

		while ( lineScanner.hasNext() )
		{
			String line = lineScanner.next();

			if(line.startsWith("#"))
				continue;

			Scanner scanner = new Scanner(line).useLocale( Locale.ENGLISH ); 
			Vector<Double> row = new Vector<Double>();
			while( scanner.hasNextDouble() )
			{
				double d = scanner.nextDouble();
				row.add( d );
			}
			data.add( row );
		}
		lineScanner.close();

		for(Vector<Double> row : data)
		{
			List<Double> fts = new Vector<Double>();
			Label label = null;
			for(int c=0;c<row.size();c++)
			{
				if(c==labelColumn)
				{
					double lab =  row.get(c).doubleValue();
					if(factor)
					{
						int ilab = (int) lab;
						label = new FactorLabel(String.format("class%d", ilab));
					}
					else
					{
						label = new NumericLabel(row.get(c).doubleValue());
					}
				}
				else
				{
					fts.add(row.get(c));
				}
			}
			
			double[] features = new double[fts.size()];
			int i=0;
			for(Double d: fts)
			{
				features[i] = d;
				i++;
			}
			PrimalInstance pi = new PrimalInstance(label,features);
			ret.add(pi);
		}
		
		int idx = 1;
		for(PrimalInstance i: ret)
		{
			i.addProperty("id", idx);
			idx++;
		}
		return ret;
	}
	
	public static void main(String[] args) throws Exception
	{

		DefaultInstanceContainer<PrimalInstance> data = readExt(new FileInputStream("/tmp/nrps.dat"), 1, true);
		
		for(PrimalInstance pi: data)
		{
			System.out.println(pi.getLabel().toString());
		}
		
		FilteredDataView<PrimalInstance> view = new FilteredDataView<PrimalInstance>();
		view.addFilter(new InstanceFilter<PrimalInstance>(){
			@Override
			public boolean accept(PrimalInstance t)
			{
				return t.getLabel().toString().equals("ala");
			}});
		view.addAll(data);
		writeExt(view, "/tmp/ala.ext");
	}
}
