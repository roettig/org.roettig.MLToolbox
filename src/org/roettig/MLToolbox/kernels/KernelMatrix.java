package org.roettig.MLToolbox.kernels;

import java.io.*;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Locale;
import java.util.Vector;

import org.roettig.MLToolbox.base.instance.Instance;
import org.roettig.MLToolbox.base.instance.InstanceContainer;
import org.roettig.MLToolbox.base.label.Label;

import Jama.Matrix;

public class KernelMatrix
{
	private Matrix matrix;
	private Vector<Label> labels;

	public static <T extends Instance> KernelMatrix compute(InstanceContainer<T> data, KernelFunction<T> f)
	{
		return compute(data, f, true);
	}

	public static <T extends Instance> KernelMatrix compute(InstanceContainer<T> data, KernelFunction<T> f, boolean normalize)
	{
		int N = data.size();
		KernelMatrix ret = new KernelMatrix(N,N);
		for(int i=0;i<N;i++)
		{
			for(int j=i;j<N;j++)
			{
				try
				{
					double val;
					if(normalize)
						val = f.computeN(data.get(i),data.get(j));
					else
						val = f.compute(data.get(i),data.get(j));
					ret.set(i,j,val);
					ret.set(j,i,val);
				} 
				catch (Exception e)
				{
					e.printStackTrace();
				}
			}
		}

		return ret;
	}

	public static <T extends Instance> KernelMatrix computeStacked(Collection<T> labdata, Collection<T> unlabdata, KernelFunction<T> f, boolean normalize)
	{
		int l = labdata.size();
		int u = unlabdata.size();
		int n = u+l;

		List<T> alldata = new ArrayList<T>();

		for(T t: labdata)
		{
			alldata.add(t);
		}
		for(T t: unlabdata)
		{
			alldata.add(t);
		}


		KernelMatrix K = new KernelMatrix(n,n);
		for(int i=0;i<n;i++)
		{
			for(int j=0;j<n;j++)
			{
				double val=0.0;
				try
				{
					if(normalize)
						val = f.computeN(alldata.get(i),alldata.get(j));
					else
						val = f.compute(alldata.get(i),alldata.get(j));
				} 
				catch (Exception e)
				{
					e.printStackTrace();
				}
				K.set(i,j,val);
			}
		}      
		return K;
	}

	public static <T extends Instance> KernelMatrix compute(InstanceContainer<T> traindata, InstanceContainer<T> testdata, KernelFunction<T> f)
	{
		return compute( traindata, testdata, f, true);
	}

	public static <T extends Instance> KernelMatrix compute(InstanceContainer<T> traindata, InstanceContainer<T> testdata, KernelFunction<T> f, boolean normalize)
	{
		KernelMatrix ret = new KernelMatrix(testdata.size(),traindata.size());
		for(int i=0;i<testdata.size();i++)
		{
			for(int j=0;j<traindata.size();j++)
			{
				try
				{
					double val;
					if(normalize)
						val = f.computeN(testdata.get(i),traindata.get(j));
					else
						val = f.compute(testdata.get(i),traindata.get(j));

					ret.set(i,j,val);
				} 
				catch (Exception e)
				{
					e.printStackTrace();
				}
			}
		}
		return ret;
	}

	public void save(String filename) throws IOException
	{
		FileWriter file_writer = new FileWriter (filename);
		BufferedWriter buf_writer = new BufferedWriter (file_writer);
		PrintWriter print_writer = new PrintWriter (buf_writer,true);
		matrix.print(print_writer,7,5);
	}

	public void saveLabelled(String filename) throws IOException
	{
		FileWriter file_writer    = new FileWriter (filename);
		BufferedWriter buf_writer = new BufferedWriter (file_writer);
		PrintWriter print_writer  = new PrintWriter (buf_writer,true);
		for(int r=0;r<matrix.getRowDimension();r++)
		{
			double[] row = this.getRow(r);
			print_writer.format(Locale.ENGLISH, "%.2f ", this.labels.get(r).getDoubleValue());
			for(int c=0;c<row.length;c++)
			{
				print_writer.format(Locale.ENGLISH, "%.5f ", this.get(r, c));   
			}
			print_writer.println("");
		}
	}

	public void saveLIBSVM(String filename) throws IOException
	{
		FileWriter file_writer    = new FileWriter (filename);
		BufferedWriter buf_writer = new BufferedWriter (file_writer);
		PrintWriter print_writer  = new PrintWriter (buf_writer,true);
		for(int r=0;r<matrix.getRowDimension();r++)
		{
			double[] row = this.getRow(r);
			print_writer.format(Locale.ENGLISH, "%.2f ", this.labels.get(r).getDoubleValue());
			print_writer.format(Locale.ENGLISH, "%d:%d ",1,r);
			for(int c=0;c<row.length;c++)
			{
				print_writer.format(Locale.ENGLISH, "%d:%.5f ",c+2, this.get(r, c));   
			}

			print_writer.println("");
		}
	}

	public void saveLIBSVMtest(String filename) throws IOException
	{
		FileWriter file_writer    = new FileWriter (filename);
		BufferedWriter buf_writer = new BufferedWriter (file_writer);
		PrintWriter print_writer  = new PrintWriter (buf_writer,true);
		for(int r=0;r<matrix.getRowDimension();r++)
		{
			double[] row = this.getRow(r);
			print_writer.format(Locale.ENGLISH, "%.2f ", this.labels.get(r).getDoubleValue());
			print_writer.format(Locale.ENGLISH, "%d:%d ",1,-1);
			for(int c=0;c<row.length;c++)
			{
				print_writer.format(Locale.ENGLISH, "%d:%.5f ",c+2, this.get(r, c));   
			}

			print_writer.println("");
		}
	}

	public KernelMatrix( double[][] A )
	{
		matrix = new Matrix(A);   
	}

	public KernelMatrix(int n, int m)
	{
		matrix = new Matrix(n,m,0.0);
	}

	public double get(int i, int j)
	{
		return matrix.get(i,j);
	}

	public void set(int i, int j, double d)
	{
		matrix.set(i,j,d);
	}

	public KernelMatrix getMatrix(int[] r, int[] c) 
	{
		Matrix tmp = matrix.getMatrix(r,c);
		return new KernelMatrix(tmp.getArrayCopy());
	}

	public void setLabels(Vector<Label> _labels)
	{
		labels = _labels;
	}

	public void setLabels(List<Label> _labels)
	{
		labels = new Vector<Label>(_labels);
	}

	public Label getLabel(int i)
	{
		return labels.get(i);
	}

	public int getRowDimension()
	{
		return matrix.getRowDimension();
	}

	public int getColumnDimension()
	{
		return matrix.getColumnDimension();
	}

	public Matrix getMatrix()
	{
		return matrix;
	}

	public double[] getRow(int i)
	{
		double[][] mat = matrix.getArrayCopy();
		return mat[i];
	}
}
