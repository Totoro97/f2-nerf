//
// Created by ppwang on 2022/9/16.
//

#pragma once
#include <memory>
#include "Field.h"
#include "../Utils/GlobalDataPool.h"

std::unique_ptr<Field> ConstructField(GlobalDataPool* global_data_pool);