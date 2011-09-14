package org.roettig.MLToolbox.model.svm;

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import org.roettig.MLToolbox.base.instance.Instance;
import org.roettig.MLToolbox.base.instance.InstanceContainer;
import org.roettig.MLToolbox.kernels.KernelFunction;
import org.roettig.MLToolbox.kernels.KernelMatrix;
import org.roettig.MLToolbox.util.libsvm.svm;
import org.roettig.MLToolbox.util.libsvm.svm_model;
import org.roettig.MLToolbox.util.libsvm.svm_node;
import org.roettig.MLToolbox.util.libsvm.svm_parameter;
import org.roettig.MLToolbox.util.libsvm.svm_problem;

public class LIBSVMDelegate<T extends Instance> implements SVMModelDelegate<T>
{

	@Override
	public void train(InstanceContainer<T> trainingdata, KernelFunction<T> k_fun, SVMTrainingParams params) throws Exception
	{
		this.k_fun = k_fun;
		this.trainingdata = trainingdata;
		
		setup( trainingdata, params);
		
		prob  = makeSVMproblem(trainingdata,k_fun, (this.param.svm_type==svm_parameter.ONE_CLASS));
    	model = null;
    	model = svm.svm_train(prob,param);
	}

	@Override
	public List<Double> predict(InstanceContainer<T> testdata)
	{
		KernelMatrix K = KernelMatrix.compute(trainingdata, testdata, k_fun, k_fun.isNormalized());
		
		List<Double> preds = new ArrayList<Double>();
		
		int Nt = testdata.size();
		for(int i=0;i<Nt;i++)
		{
			if(param.svm_type==svm_parameter.ONE_CLASS)
			{
				double val[] = new double[1];
				svm.svm_predict_values(model, makeSVMnode(K.getRow(i),0), val);
				double yp = val[0];
				preds.add(yp);
			}
			else
			{
				double yp = svm.svm_predict(model, makeSVMnode(K.getRow(i),0));
				preds.add(yp);
			}
		}
		
		return preds;
	}

	protected svm_parameter param;
	protected svm_problem   prob;
	protected svm_model     model;
    protected InstanceContainer<T> trainingdata;    
    protected KernelFunction<T> k_fun;
    
    public LIBSVMDelegate()
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
    }
    
    private boolean validConfig;
    
    private void setup(InstanceContainer<T> trainingdata, SVMTrainingParams params)
    {
    	validConfig = false;
    	
    	if(trainingdata.isFactorLabelled())
    	{
    		if(params.hasKey(SVMTrainingParams.C))
    		{
    			param.svm_type    = svm_parameter.C_SVC;
    			param.C           = params.getValue(SVMTrainingParams.C); 
    		}
    		else if(params.hasKey(SVMTrainingParams.NU))
    		{
    			param.svm_type    = svm_parameter.NU_SVC;
    			param.nu          = params.getValue(SVMTrainingParams.NU);
    		}
    		else
    		{
    			validConfig = false;
    		}
    		
    		if(params.getFlag(SVMTrainingParams.OneClass))
    		{
    			param.svm_type    = svm_parameter.ONE_CLASS;
    		}
    	}
    	else
    	{
    		if(params.hasKey(SVMTrainingParams.C))
    		{
    			
    			param.C           = params.getValue(SVMTrainingParams.C);
    			
    			if(params.hasKey(SVMTrainingParams.EPS))
    			{
    				param.svm_type    = svm_parameter.EPSILON_SVR;
    				param.eps  =  params.getValue(SVMTrainingParams.EPS);
    			}
    			else if(params.hasKey(SVMTrainingParams.NU))
    			{
    				param.svm_type    = svm_parameter.NU_SVR;
    				param.nu   =  params.getValue(SVMTrainingParams.NU);
    			}
    			else
    			{
    				validConfig = false;
    			}
    		}
    		else
    		{
    			validConfig = false;
    		}
    	}
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

	@Override
	public double getObjectiveValue(int i)
	{
		return model.obj[i];
	}
}
