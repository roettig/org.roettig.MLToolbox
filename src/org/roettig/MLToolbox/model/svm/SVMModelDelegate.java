package org.roettig.MLToolbox.model.svm;

import java.io.Serializable;
import java.util.List;

import org.roettig.MLToolbox.base.instance.Instance;
import org.roettig.MLToolbox.base.instance.InstanceContainer;
import org.roettig.MLToolbox.kernels.KernelFunction;

public interface SVMModelDelegate<T extends Instance> extends Serializable
{
	void train(InstanceContainer<T> trainingdata, KernelFunction<T> k_fun, SVMTrainingParams params) throws Exception;
	List<Double> predict(InstanceContainer<T> trainingdata);
	public double getObjectiveValue(int i);
}
