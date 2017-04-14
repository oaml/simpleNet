#ifndef  NEURAL_H
#define  NEURAL_H

#include <iostream>
#include <cassert>
#include <vector>
#include "neuron.h"
#include "layer.h"
#include "matrix.h"
#include <algorithm>
#include <typeinfo>
#include <memory>
#include <fstream>
#include <time.h>


//enum hiddenType  {relu, sigmoid};
//enum outputType  {softmax, sigmoidOut};


template <class NType, class LType>
class NeuralNetwork
{

    unsigned int inputLayerSize, layerNumber;
    std::vector<std::unique_ptr<HiddenLayer<NType> > > HiddenLayers;
    std::unique_ptr<OutputLayer<NType> > OutLayer;
    bool outputLayerAdded;
    double learningRate;


    public:
    NeuralNetwork(const unsigned int &inputLayerSize, const double &learningRate)
    {
        srand(time(NULL));
        this->learningRate = learningRate;
        this->inputLayerSize = inputLayerSize;
        this->layerNumber = 0;
        outputLayerAdded = false;
    }

    void addHiddenLayer(const unsigned int &layerSize, hiddenType type)
    {
        assert(!outputLayerAdded);

        switch(type)
        {
            case relu:
                if(layerNumber == 0)

                    HiddenLayers.emplace_back(new LayerRelu<NType>(layerSize, inputLayerSize, relu));
                else
                {
                    HiddenLayers.emplace_back(new LayerRelu<NType>(layerSize, HiddenLayers[layerNumber - 1]->getSize(), relu));
                }
                layerNumber++;
                break;
			case sigmoid:
				if(layerNumber == 0)

                    HiddenLayers.emplace_back(new LayerSigmoid<NType>(layerSize, inputLayerSize, sigmoid));
                else
                {
                    HiddenLayers.emplace_back(new LayerSigmoid<NType>(layerSize, HiddenLayers[layerNumber - 1]->getSize(), sigmoid));
                }
                layerNumber++;
				break;
            default:
                std::cerr << "hidden layer type not implemented";
        }
    }

    void addOutputLayer(const unsigned int &layerSize, outputType type)
    {
        assert(!outputLayerAdded);
        switch(type)
        {
            case softmax:
                OutLayer.reset(new LayerSoftmax<NType>(layerSize, HiddenLayers[layerNumber - 1]->getSize(), softmax));
                outputLayerAdded = true;
                layerNumber++;
				updateLayerSizes();
                break;
            case sigmoidOut:
                OutLayer.reset(new LayerSigmoidOut<NType>(layerSize, HiddenLayers[layerNumber - 1]->getSize(), sigmoidOut));
                outputLayerAdded = true;
                layerNumber++;
				updateLayerSizes();
                break;
            default:
                std::cerr << "output layer type not implemented";
        }
    }

	void updateLayerSizes()
	{
		for(int i = 0; i < HiddenLayers.size() - 1;i++)
		{
			HiddenLayers[i]->setForwardSize(HiddenLayers[i + 1]->getSize());
			HiddenLayers[i]->generateNeurons();
		}
		HiddenLayers[HiddenLayers.size() - 1]->setForwardSize(OutLayer->getSize());
		HiddenLayers[HiddenLayers.size() - 1]->generateNeurons();
		OutLayer->setForwardSize(0);
		OutLayer->generateNeurons();
	}

    Matrix<NType> feedForward(const std::vector<NType> &inputVector, std::vector<Matrix<NType> > *keepOutputsVector = NULL) __attribute__ ((hot))
    {
        assert(outputLayerAdded);
        assert(inputVector.size() == inputLayerSize);
        Matrix<NType> inputMatrix(1, inputVector.size(), inputVector);
        if(keepOutputsVector != NULL)
        {
            keepOutputsVector->push_back(inputMatrix);
        }
        for(int currentLayer = 0; currentLayer < layerNumber;currentLayer++)
        {
            if(currentLayer != layerNumber - 1)
            {
                inputMatrix = inputMatrix * HiddenLayers[currentLayer]->getWeightMatrix(true);

                HiddenLayers[currentLayer]->applyActivationFunction(inputMatrix);
            }
            else
            {
                inputMatrix = inputMatrix * OutLayer->getWeightMatrix(true);
                OutLayer->applyActivationFunction(inputMatrix);
            }
            if(keepOutputsVector != NULL)
            {
                keepOutputsVector->push_back(inputMatrix);
            }

        }
        Matrix<NType> w(inputMatrix);
        return w;
    }

     double backpropBatch(const std::vector<std::vector<NType> > &inputs, const std::vector<std::vector<LType> > &labels) __attribute__ ((hot))
     {
         assert(outputLayerAdded);
         assert(inputs.size() == labels.size());
         std::vector<Matrix<NType> > Outputs;
         std::vector<Matrix<NType> > WeightUpdates;
		 std::vector<Matrix<NType> > BiasUpdates;
         double totalError = 0;
         for(int i = 0;i < HiddenLayers.size();i++)
         {
                 WeightUpdates.push_back(Matrix<NType> (HiddenLayers[i]->getSize(), HiddenLayers[i]->getPreviousSize()));
				 BiasUpdates.push_back(Matrix<NType> (1, HiddenLayers[i]->getSize()));
         }
         WeightUpdates.push_back(Matrix<NType> (OutLayer->getSize(), OutLayer->getPreviousSize()));
         BiasUpdates.push_back(Matrix<NType> (1, OutLayer->getSize()));
         for(int currentInput = 0;currentInput < inputs.size();currentInput++)
         {
             assert(labels[currentInput].size() == OutLayer->getSize());
             assert(inputs[currentInput].size() == inputLayerSize);
             feedForward(inputs[currentInput], &Outputs);
             Matrix<NType> DeltaMatrix = OutLayer->getOutputDeltaMatrix(Outputs[layerNumber], labels[currentInput]);
			 BiasUpdates[layerNumber - 1] = BiasUpdates[layerNumber - 1] + DeltaMatrix;
			 WeightUpdates[layerNumber - 1] = WeightUpdates[layerNumber - 1] + hardToNameThis(DeltaMatrix, Outputs[layerNumber - 1]);
             DeltaMatrix = DeltaMatrix * OutLayer->getWeightMatrix(true).transpose();
             for(int currentLayer = layerNumber - 2;currentLayer >= 0;currentLayer--)
              {
                 DeltaMatrix = HiddenLayers[currentLayer]->applyActivationDerivative(Outputs[currentLayer + 1], DeltaMatrix);
                 BiasUpdates[currentLayer] = BiasUpdates[currentLayer] + DeltaMatrix;
				 WeightUpdates[currentLayer] = WeightUpdates[currentLayer] + hardToNameThis(DeltaMatrix, Outputs[currentLayer]);
                 DeltaMatrix = DeltaMatrix * HiddenLayers[currentLayer]->getWeightMatrix(true).transpose();
             }
             totalError += OutLayer->getError();
             Outputs.clear();
         }

         totalError /= (double)inputs.size();
         for(int i = 0;i < WeightUpdates.size();i++)
         {
             WeightUpdates[i] *  learningRate;
			 BiasUpdates[i] *  learningRate;
         }
         for(int i = 0;i < HiddenLayers.size();i++)
         {
             HiddenLayers[i]->updateLayerWeights(WeightUpdates[i]);
			 HiddenLayers[i]->updateLayerBias(BiasUpdates[i]);
         }
         OutLayer->updateLayerWeights(WeightUpdates[layerNumber - 1]);
		 OutLayer->updateLayerBias(BiasUpdates[layerNumber - 1]);
		 WeightUpdates.clear();
         BiasUpdates.clear();
		 return totalError;
     }

     friend inline std::ofstream & operator <<(std::ofstream &out, const NeuralNetwork<NType, LType> &n)
     {
         if(!out.is_open() || !n.outputLayerAdded)
            return out;
         out << n.layerNumber << std::endl;
         out << n.inputLayerSize << std::endl;
         out << n.learningRate << std::endl;
         for(int i = 0;i < n.HiddenLayers.size();i++)
         {
             out << n.HiddenLayers[i]->getSize() << std::endl;
             out << n.HiddenLayers[i]->getPreviousSize() << std::endl;
             out << n.HiddenLayers[i]->getType() << std::endl;
             out << n.HiddenLayers[i]->getForwardSize() << std::endl;
             std::vector<NType> layerData = n.HiddenLayers[i]->getWeightMatrix().getData();
             for(int j = 0;j < layerData.size();j++)
             {
                 out << layerData[j] << '\n';
             }
             std::vector<NType> biasData = n.HiddenLayers[i]->getBiasVector();
             for(int j = 0;j < biasData.size();j++)
             {
                 out << biasData[j] << '\n';
             }
         }
         out << n.OutLayer->getSize() << std::endl;
         out << n.OutLayer->getPreviousSize() << std::endl;
         out << n.OutLayer->getType() << std::endl;
         out << n.OutLayer->getForwardSize() << std::endl;
         std::vector<NType> layerData = n.OutLayer->getWeightMatrix().getData();
         for(int j = 0;j < layerData.size();j++)
         {
             out << layerData[j] << '\n';
         }
         std::vector<NType> biasData = n.OutLayer->getBiasVector();
         for(int j = 0;j < biasData.size();j++)
         {
             out << biasData[j] << '\n';
         }
         return out;
     }

     friend inline std::ifstream & operator >>(std::ifstream &in, NeuralNetwork<NType, LType> &n)
     {
         if(!in.is_open())
            return in;
         n.HiddenLayers.clear();
         in >> n.layerNumber;
         in >> n.inputLayerSize;
         in >> n.learningRate;
         unsigned int currentSize;
         unsigned int currentPreviousSize;
         unsigned int temp;
         unsigned int currentForwardSize;
         NType weight;
         NType bias;
         hiddenType currentHiddenType;
         outputType currentOutputType;
         std::vector<NType> biasData;
         std::vector<std::vector<NType>> weightData;
         std::vector<Neuron<NType> > LayerNeurons;
         for(int i = 0;i < n.layerNumber - 1;i++)
         {
             in >> currentSize;
             in >> currentPreviousSize;
             in >> temp;
             in >> currentForwardSize;
             currentHiddenType = (hiddenType)temp;
             for(int k = 0;k < currentSize;k++)
             {
                 weightData.push_back(std::vector<NType>());
                 for(int j = 0;j < currentPreviousSize;j++)
                 {
                     in >> weight;
                     weightData[k].push_back(weight);
                 }
             }
             biasData.reserve(currentSize);
             for(int j = 0;j < currentSize;j++)
             {
                 in >> bias;
                 biasData.push_back(bias);
             }
             for(int j = 0;j < currentSize;j++)
             {
                 LayerNeurons.push_back(Neuron<NType>(currentPreviousSize, currentForwardSize, &(weightData[j]), biasData[j]));
             }
             switch(currentHiddenType)
             {
                case relu:
                    n.HiddenLayers.emplace_back(new LayerRelu<NType>(currentSize, currentPreviousSize, currentHiddenType, currentForwardSize, &LayerNeurons));
                    break;
                case sigmoid:
                    n.HiddenLayers.emplace_back(new LayerSigmoid<NType>(currentSize, currentPreviousSize, currentHiddenType, currentForwardSize, &LayerNeurons));
                    break;
             }
             LayerNeurons.clear();
             biasData.clear();
             weightData.clear();
         }
         in >> currentSize;
         in >> currentPreviousSize;
         in >> temp;
         in >> currentForwardSize;
         currentOutputType = (outputType)temp;
         for(int k = 0;k < currentSize;k++)
         {
                 weightData.push_back(std::vector<NType>());
                 for(int j = 0;j < currentPreviousSize;j++)
                 {
                     in >> weight;
                     weightData[k].push_back(weight);
                 }
             }
         biasData.reserve(currentSize);
             for(int j = 0;j < currentSize;j++)
             {
                 in >> bias;
                 biasData.push_back(bias);
             }
             for(int j = 0;j < currentSize;j++)
             {
                 LayerNeurons.push_back(Neuron<NType>(currentPreviousSize, currentForwardSize, &(weightData[j]), biasData[j]));
             }
         switch(currentOutputType)
         {
            case sigmoidOut:
                n.OutLayer.reset(new LayerSigmoidOut<NType>(currentSize, currentPreviousSize, currentOutputType, currentForwardSize, &LayerNeurons));
                break;
            case softmax:
                n.OutLayer.reset(new LayerSoftmax<NType>(currentSize, currentPreviousSize, currentOutputType, currentForwardSize, &LayerNeurons));
                break;
         }
         n.outputLayerAdded = true;
         return in;
     }


};

#endif


