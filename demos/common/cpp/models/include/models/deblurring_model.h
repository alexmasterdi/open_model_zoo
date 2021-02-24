/*
// Copyright (C) 2018-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writingb  software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "model_base.h"
#include "opencv2/core.hpp"

#pragma once
class DeblurringModel : public ImageProcessingModel {
public:
    /// Constructor
    /// @param modelFileName name of model to load
    /// Otherwise, image will be preprocessed and resized using OpenCV routines.
    DeblurringModel(const std::string& modelFileName, const cv::Size& inputImageShape);

    std::shared_ptr<InternalModelData> preprocess(
        const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) override;

    std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) override;
    void reshape(InferenceEngine::CNNNetwork & cnnNetwork) override;
protected:
    void prepareInputsOutputs(InferenceEngine::CNNNetwork & cnnNetwork) override;
    size_t blockSize = 32;
    int inputHeight = 0;
    int inputWidth = 0;
    int outHeight = 0;
    int outWidth = 0;
};
