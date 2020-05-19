#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>

int main() {
    auto mod = torch::jit::load("resnet18.pt");
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::randn({1, 3, 224, 224}));
    std::cout<<mod.forward(inputs).toTensor().argmax(1)<<std::endl;
    return 0;
}
