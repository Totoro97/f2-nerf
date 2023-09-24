#include <iostream>
#include <memory>
#include <torch/torch.h>
#include "src/ExpRunner.h"

int main(int argc, char *argv[])
{
  std::cout << "Aoligei!" << std::endl;
  torch::manual_seed(2022);

  std::string conf_path = "./runtime_config.yaml";
  auto exp_runner = std::make_unique<ExpRunner>(conf_path);
  exp_runner->Execute();
  return 0;
}
