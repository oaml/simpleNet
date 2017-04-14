#include "neuralnetwork.h"

#include <iostream>

#include <fstream>
#include <vector>


using namespace std;

int main()
{
    NeuralNetwork<double, double> net(2, 0.01);
    net.addHiddenLayer(2, sigmoid);
    net.addHiddenLayer(2, relu);
    net.addOutputLayer(2, softmax);
    vector<vector<double> > inputs;
    vector<double> input;
    input.push_back(1);
    input.push_back(0);
    inputs.push_back(input);
    vector<vector<double> > outputs;
    vector<double> output;
    output.push_back(0);
    output.push_back(1);
    outputs.push_back(output);
    for(int i = 0;i < 100;i++) {
        net.backpropBatch(inputs, outputs);
    }
    ofstream out;
    out.open("neural.net", ios_base::out);
    out << net;
    return 0;
}
