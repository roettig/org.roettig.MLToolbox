package org.roettig.MLToolbox.base;

public interface PrimalEncoder<T>
{
	double[] encode(T t);
}