package org.roettig.MLToolbox.model.svm;

public interface SVMTrainingParams
{
	public boolean hasKey(String key);
	public void    setValue(String key, double value);
	public double  getValue(String key);
	public void    setFlag(String key);
	public boolean getFlag(String key);
	
	static String NU       = "NU";
	static String C        = "C";
	static String EPS      = "EPS";
	static String OneClass = "OneClass";
	static String Transductive = "Transductive";
}
