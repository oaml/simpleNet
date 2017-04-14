#ifndef  NEURON_H
#define  NEURON_H


#include <stdlib.h>
#include <time.h>
#include <vector>
#include <iostream>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <random>

static int count = 0;

template<class NType>
class Neuron
{
	std::vector<NType> weights;
	double bias;

	public:
	Neuron(const unsigned int &previousSize, const unsigned int &forwardSize, std::vector<NType> *weightData = NULL, NType bias = -1)
	{
		if(weightData != NULL && bias != -1)
		{
		    weights = *weightData;
		    this->bias = bias;
		}
        else
        {
            auto engine = std::default_random_engine{};
            std::normal_distribution<double> distribution(0, 0.15);
            //std::normal_distribution<double> distribution(0, sqrt(6 / (previousSize + forwardSize)));
            //std::uniform_real_distribution<double> distribution(-4*sqrt(6 / (NType)(previousSize + forwardSize)), 4*sqrt(6 / (NType)(previousSize + forwardSize)));
            for(int i = 0;i < previousSize;i++)
            {
                //TODO:
                //According to Hugo Larochelle, Glorot & Bengio (2010),
                //initialize the weights uniformly within the interval [−b,b],
                //where b=(6/(Hk+Hk+1))^(1/2), Hk and Hk+1 are the sizes of the
                //layers before and after the weight matrix.
                weights.push_back(distribution(engine));
            }
            this->bias = distribution(engine);
			count++;
        }
	}

	~Neuron()
	{
		weights.clear();
	}
	
	static int getCount()
	{
		return count;
	}

	const std::vector<NType> &getWeights()
	{
		return weights;
	}

	NType getBias()
	{
		return bias;
	}

	void updateBias(const NType &bias)
	{
		this->bias -= bias;
	}

	void updateWeights(const std::vector<NType> &weightDeltas)
	{
		//std::cout << weightDeltas.size() << std::endl << weights.size() << std::endl;
	    assert(weights.size() == weightDeltas.size());
	    for(int i = 0;i < weights.size();i++)
        {
			//std::cout << weights[i] << std::endl;
            weights[i] -= weightDeltas[i];
			//std::cout << weights[i];
		}
		//std::cout << std::endl;
	}
};

#endif

