/*
// Copyright (C) 2018-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "models/deblurring_model.h"
#include "utils/ocv_common.hpp"

#include <iostream>

using namespace InferenceEngine;

DeblurringModel::DeblurringModel(const std::string& modelFileName, const cv::Size& inputImageSize) :
    ModelBase(modelFileName), inputHeight(inputImageSize.height), inputWidth(inputImageSize.width) {
}

void DeblurringModel::reshape(InferenceEngine::CNNNetwork & cnnNetwork) {
    auto shapes = cnnNetwork.getInputShapes();
    size_t blockSize = 32;
    for (auto& shape : shapes) {
        shape.second[0] = 1;
        shape.second[2] = (inputHeight + blockSize - 1)/blockSize * blockSize;
        shape.second[3] = (inputWidth + blockSize - 1)/blockSize * blockSize;
    }
    cnnNetwork.reshape(shapes);
}

void DeblurringModel::prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) {
    // --------------------------- Configure input & output ---------------------------------------------
    // --------------------------- Prepare input blobs --------------------------------------------------
    ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    if (inputShapes.size() != 1 && inputShapes.size() != 2)
        throw std::logic_error("The demo supports topologies with 1 or 2 inputs only");
    std::string firstInputBlobName = inputShapes.begin()->first;
    inputsNames.push_back(firstInputBlobName);
    SizeVector& firstInputSizeVector = inputShapes[firstInputBlobName];
    if (firstInputSizeVector.size() != 4)
        throw std::logic_error("Number of dimensions for an input must be 4");
    std::cout << firstInputSizeVector[2] << '\n';
    InputInfo& inputInfo = *cnnNetwork.getInputsInfo().begin()->second;
    inputInfo.setPrecision(Precision::FP32);
    // --------------------------- Prepare output blobs -----------------------------------------------------
    const OutputsDataMap& outputInfo = cnnNetwork.getOutputsInfo();
    if (outputInfo.size() != 1) throw std::runtime_error("Demo supports topologies only with 1 output");

    outputsNames.push_back(outputInfo.begin()->first);
    Data& data = *outputInfo.begin()->second;
    data.setPrecision(Precision::FP32);
    const SizeVector& outSizeVector = data.getTensorDesc().getDims();
    outHeight = outSizeVector[2];
    outWidth = outSizeVector[3];
}

std::shared_ptr<InternalModelData> DeblurringModel::preprocess(const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) {
    auto imgData = inputData.asRef<ImageInputData>();
    auto& img = imgData.inputImage;

    /* Padding and copy data from the image to the input blob */
    Blob::Ptr inputBlob = request->GetBlob(inputsNames[0]);
    int w = inputBlob->getTensorDesc().getDims()[3];
    int h = inputBlob->getTensorDesc().getDims()[2];

    cv::Mat resized;
    if (img.rows < h && img.cols < w)
        cv::copyMakeBorder(img, resized, 0, h - img.rows, 0, w - img.cols, cv::BORDER_CONSTANT, 0);
    else
        cv::resize(img, resized, cv::Size(w, h), 0, 0, cv::INTER_CUBIC);

    matU8ToBlob<float_t>(resized, inputBlob);

    return std::shared_ptr<InternalModelData>(new InternalImageModelData(resized.cols, resized.rows));
}

std::unique_ptr<ResultBase> DeblurringModel::postprocess(InferenceResult& infResult) {
    ImageProcessingResult* result = new ImageProcessingResult;
    *static_cast<ResultBase*>(result) = static_cast<ResultBase&>(infResult);

    LockedMemory<const void> outMapped = infResult.getFirstOutputBlob()->rmap();
    const auto outputData = outMapped.as<float*>();

    std::vector<cv::Mat> imgPlanes;
    size_t numOfPixels = outWidth * outHeight;
    // for (int i = numOfPixels; i < 2 * numOfPixels; ++i) {
    //     std::cout << outputData[i] << " ";
    // }
    imgPlanes = std::vector<cv::Mat>{
        cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[0])),
        cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[numOfPixels])),
        cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[numOfPixels * 2]))};

    cv::merge(imgPlanes, result->resultImage);
    result->resultImage = result->resultImage(cv::Rect(0, 0, std::min(inputWidth,result->resultImage.cols), std::min(inputHeight, result->resultImage.rows)));
    result->resultImage.convertTo(result->resultImage, CV_8UC3, 255);

    return std::unique_ptr<ResultBase>(result);
}
