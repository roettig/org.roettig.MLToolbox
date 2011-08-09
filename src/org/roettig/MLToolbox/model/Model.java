package org.roettig.MLToolbox.model;


import java.util.List;

import org.roettig.MLToolbox.base.Prediction;
import org.roettig.MLToolbox.base.impl.DefaultParametrizedComposite;
import org.roettig.MLToolbox.base.instance.Instance;
import org.roettig.MLToolbox.base.instance.InstanceContainer;
import org.roettig.MLToolbox.validation.QualityMeasure;

public abstract class Model<T extends Instance> extends DefaultParametrizedComposite
{
	public abstract void train(InstanceContainer<T> trainingdata);
	public abstract List<Prediction> predict(InstanceContainer<T> testdata);
	public abstract double getQuality(List<Prediction> predictions);
	public abstract void setQualityMeasure(QualityMeasure qm);
}
