/**
 * 
 */
package org.roettig.MLToolbox.util.libsvm;


import java.io.Serializable;
import java.util.Vector;

import org.roettig.MLToolbox.base.instance.Instance;
import org.roettig.MLToolbox.base.instance.InstanceContainer;
import org.roettig.MLToolbox.kernels.KernelFunction;
import org.roettig.MLToolbox.kernels.KernelMatrix;


/**
 * @author roettig
 *
 */
public class libsvmDelegate<T extends Instance> implements Serializable
{
    public svm_parameter param;
    public svm_problem   prob;
    public svm_model     model;
        
    protected KernelFunction<T> k_fun;
    
    public libsvmDelegate(KernelFunction<T> _k_fun)
    {
    	param = new svm_parameter();
        // default values
        param.svm_type    = svm_parameter.C_SVC;
        param.kernel_type = svm_parameter.PRECOMPUTED;
        param.degree = 3;
        param.gamma = 0;    // 1/k
        param.coef0 = 0;
        param.nu = 0.5;
        param.cache_size = 100;
        param.C = 100;
        param.eps = 1e-3;
        param.p = 0.1;
        param.shrinking = 1;
        param.probability = 0;
        param.nr_weight = 0;
        param.weight_label = new int[0];
        param.weight = new double[0];
        k_fun = _k_fun;
    }
    
    public void train(InstanceContainer<T> trainingdata)
    {
    	prob  = makeSVMproblem(trainingdata,k_fun, (this.param.svm_type==svm_parameter.ONE_CLASS));
    	model = null;
    	model = svm.svm_train(prob,param);
    }

    @Override
    public libsvmDelegate<T> clone()
    {
    	/*
    	libsvmDelegate<T> ret = null;
    	ret = (libsvmDelegate<T>) clone();
    	ret.k_fun = k_fun.clone();
    	if(param!=null)
    		ret.param = (svm_parameter) param.clone();
    	if(prob!=null)
    		ret.prob  = clone_svm_prob(prob);
    	if(model!=null)
    		ret.model = clone_svm_model(model);
		
    	return ret;
    	*/
    	return null;
    }

    public static <T extends Instance> svm_problem makeSVMproblem(InstanceContainer<T> data, KernelFunction<T> k_fun, boolean one_class)
    {
    	KernelMatrix K   = KernelMatrix.compute(data, k_fun, k_fun.isNormalized());

    	svm_problem  ret = new svm_problem();

    	Vector<svm_node[]> vx = new Vector<svm_node[]>();

    	// get sample size
    	int n = K.getRowDimension();

    	// create for each row in KernelMatrix a svm_node-list
    	for(int i=0;i<n;i++)
    	{
    		double[] row = K.getRow(i);
    		svm_node[] x = makeSVMnode(row,i+1);  
    		vx.addElement(x);
    	}

    	// set number of training samples
    	ret.l = n;
    	// create x-array
    	ret.x = new svm_node[ret.l][];
    	// copy svm_node-lists into problem
    	for(int i=0;i<ret.l;i++)
    		ret.x[i] = vx.elementAt(i);
    	// create new label array (y-array)
    	ret.y = new double[ret.l];
    	// copy class labels int y-array
    	for(int i=0;i<ret.l;i++)
    	{
    		
    		//if(data.getLabelSupplier()!=null)
    		//{
    		//	ret.y[i] = data.getLabelSupplier().getLabel(i).getDoubleValue();
    		//}
    		if(one_class)
    		{
    			ret.y[i] = 1.0;
    		}
    		else
    		{
    			ret.y[i] = data.get(i).getLabel().getDoubleValue();
    		}
    	}
    	return ret;
    }

    public static svm_node[] makeSVMnode(double[] row, int idx)
    {
    	int m        = row.length;
    	svm_node[] x = new svm_node[m+1];

    	// PRECOMPUTED svm_node lists have leading sample IDX
    	x[0]         = new svm_node();
    	x[0].index   = 0; 
    	x[0].value   = idx;

    	for(int j=1;j<=m;j++)
    	{
    		x[j]       = new svm_node();
    		x[j].index = j; 
    		x[j].value = row[j-1];
    	}
    	return x;
    }

    public static svm_problem clone_svm_prob(svm_problem p)
    {
    	svm_problem ret = new svm_problem();
    	if(p.x!=null)
    		ret.x = p.x.clone();
    	if(p.y!=null)
    		ret.y = p.y.clone();
    	ret.l = p.l;
    	return ret;
    }

    public static svm_model clone_svm_model(svm_model m)
    {
    	svm_model ret = new svm_model();
    	ret.l = m.l;
    	ret.label = m.label.clone();
    	ret.nr_class = m.nr_class;
    	ret.nSV = m.nSV.clone();
    	ret.param = (svm_parameter) m.param.clone();
    	ret.probA = m.probA.clone();
    	ret.probB = m.probB.clone();
    	ret.rho = m.rho.clone();
    	ret.SV  = m.SV.clone();
    	ret.sv_coef = m.sv_coef.clone();
    	return ret;
    }

    public static void main(String[] args)
    {


    }

}
