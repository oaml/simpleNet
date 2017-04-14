#ifndef  LAYER_H
#define  LAYER_H


#include <vector>
#include "neuron.h"
#include "matrix.h"
#include <iostream>
#include <cmath>

enum hiddenType  {relu, sigmoid};
enum outputType  {softmax, sigmoidOut};


template<class NType>
class Layer
{
	unsigned int size, previousSize, forwardSize;
    std::vector<Neuron<NType> > neurons;

	public:
	explicit Layer(const unsigned int &size, const unsigned int &previousSize, const int forwardSize = -1, std::vector<Neuron<NType>> *neurons = NULL)
	{

		this->size = size;
		this->previousSize = previousSize;
		if(neurons != NULL)
            this->neurons = *neurons;
        if(forwardSize >= 0)
            this->forwardSize = forwardSize;
	}

	unsigned int getSize()
	{
	    return size;
	}

	unsigned int getPreviousSize()
	{
	    return previousSize;
	}

	void setForwardSize(const unsigned int &n)
	{
		forwardSize = n;
	}

	unsigned int getForwardSize()
	{
	    return forwardSize;
	}

	void generateNeurons()
	{
		for(int i = 0;i < size;i++)
		{
			neurons.push_back(Neuron<NType>(previousSize, forwardSize));
		}
	}


	std::vector<NType> getBiasVector()
	{
		std::vector<NType> biasVector;
		for(int i = 0;i < size;i++)
		{
			biasVector.push_back(neurons[i].getBias());
		}
		return biasVector;
	}

	Matrix<NType> getWeightMatrix(bool columnMajor = false)
	{
	    std::vector<NType> totalWeights;
	    std::vector<NType> tempVector;
	    for(int i = 0;i < neurons.size();i++)
        {
            tempVector = neurons[i].getWeights();
            totalWeights.insert(totalWeights.end(), tempVector.begin(), tempVector.end());
        }
        Matrix<NType> WeightMatrix(previousSize, size, totalWeights, columnMajor);
        return WeightMatrix;
	}

	void updateLayerWeights(const Matrix<NType> &weightDeltas)
	{
	    for(int i = 0;i < neurons.size();i++)
	    {
	        neurons[i].updateWeights(weightDeltas.getRowData(i));
	    }
	}

	void updateLayerBias(const Matrix<NType> &biasDeltas)
	{
		std::vector<NType> data = biasDeltas.getData();
		assert(data.size() == size);
		for(int i = 0;i < size;i++)
		{
			neurons[i].updateBias(data[i]);
		}
	}


	virtual void applyActivationFunction(Matrix<NType> &inputs) = 0;
};

template<class NType>
class HiddenLayer : public Layer<NType>
{
    hiddenType type;

    public:
    HiddenLayer(const unsigned int &size, const unsigned int &previousSize, hiddenType type, int forwardSize = -1, std::vector<Neuron<NType>> *neurons = NULL)
    : Layer<NType>(size, previousSize, forwardSize, neurons)
    {
        this->type = type;
    }

    hiddenType getType() 
    {
        return type;
    }
	
    virtual Matrix<NType> applyActivationDerivative(const Matrix<NType> &Outputs, Matrix<NType> DeltaMatrix) = 0;


};

template<class NType>
class OutputLayer : public Layer<NType>
{
    outputType type;

    protected:
    double error;

    public:
    OutputLayer(const unsigned int &size, const unsigned int &previousSize, outputType type, int forwardSize = -1, std::vector<Neuron<NType>> *neurons = NULL)
    : Layer<NType>(size, previousSize, forwardSize, neurons)
    {
        this->type = type;
    }

    double getError()
    {
        return error;
    }

    outputType getType()
    {
        return type;
    }

    virtual Matrix<NType> getOutputDeltaMatrix(const Matrix<NType> &outputs, const std::vector<NType> &labels) = 0;

};


template<class NType>
class LayerRelu : public HiddenLayer<NType>
{

	public:
	using HiddenLayer<NType>::HiddenLayer;

    void applyActivationFunction(Matrix<NType> &inputs)
    {
		std::vector<NType> biasVector = this->getBiasVector();
		inputs + biasVector;
        inputs.applyFunction([] (NType a) {return std::max((NType)0, a);});
    }

    Matrix<NType> applyActivationDerivative(const Matrix<NType> &Outputs, const Matrix<NType> DeltaMatrix)
    {
		DeltaMatrix.reluActivation(Outputs);
        return DeltaMatrix;
    }

};

template<class NType>
class LayerSigmoid : public HiddenLayer<NType>
{
    public:
	using HiddenLayer<NType>::HiddenLayer;
	void applyActivationFunction(Matrix<NType> &inputs)
    {
		std::vector<NType> biasVector = this->getBiasVector();
		inputs + biasVector;
        inputs.applyFunction([] (NType a) {return 1/(1+exp(-a));});
    }

	Matrix<NType> applyActivationDerivative(const Matrix<NType> &Outputs, const Matrix<NType> DeltaMatrix)
    {
		DeltaMatrix.sigmoidActivation(Outputs);
        return DeltaMatrix;
    }
};



template<class NType>
class LayerSigmoidOut : public OutputLayer<NType>
{
    public:
    using OutputLayer<NType>::OutputLayer;
    void applyActivationFunction(Matrix<NType> &inputs)
    {
		std::vector<NType> biasVector = this->getBiasVector();
		inputs + biasVector;
        inputs.applyFunction([] (NType a) {return 1/(1+exp(-a));});
    }

    Matrix<NType> getOutputDeltaMatrix(const Matrix<NType> &outputs, const std::vector<NType> &labels)
    {
        this->error = 0;
        std::vector<NType> outputVec = outputs.getData();
        std::vector<NType> deltasVec;
        for(int i = 0;i < outputVec.size();i++)
        {
            deltasVec.push_back( -(labels[i] - outputVec[i]) * outputVec[i] * (1 - outputVec[i]));
            this->error += ((labels[i] - outputVec[i]) * (labels[i] - outputVec[i])) / 2;
        }
        Matrix<NType> DeltaMatrix(1, deltasVec.size() ,deltasVec);
        return DeltaMatrix;
    }


};



template<class NType>
class LayerSoftmax : public OutputLayer<NType>
{


    public:
    using OutputLayer<NType>::OutputLayer;

    void applyActivationFunction(Matrix<NType> &inputs)
    {
        std::vector<NType> biasVector = this->getBiasVector();
		inputs + biasVector;
        NType exponentialSum = 0;
        std::vector<NType> inputVec = inputs.getData();
        for(int i = 0;i < inputVec.size();i++)
        {
            exponentialSum += exp(inputVec[i]);
        }
        inputs.softmaxOutputActivation(exponentialSum);

    }

    Matrix<NType> getOutputDeltaMatrix(const Matrix<NType> &outputs, const std::vector<NType> &labels)
    {
        std::vector<NType> outputVec = outputs.getData();
        int index = -1;
        for(int i = 0;i < labels.size();i++)
        {
            if(labels[i] == 1)
            {
                index = i;
                break;
            }
        }
        assert(index > -1);
        this->error = log(outputVec[index]) * (-1);
        outputVec[index] -= 1;
        Matrix<NType> DeltaMatrix(1, outputs.getColumns(), outputVec);
        return DeltaMatrix;
    }


};


#endif

