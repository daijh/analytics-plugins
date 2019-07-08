// Copyright (C) <2019> Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#ifndef DEEPLAB_H
#define DEEPLAB_H

#include <inference_engine.hpp>

using namespace InferenceEngine;

class Deeplab {

public:
    Deeplab(std::string device = "CPU");
    virtual ~Deeplab();

    bool init();

    bool infer(cv::Mat& image, cv::Mat *segmentation);

private:
    std::string m_device;
    std::string m_model;

    InferencePlugin m_plugin;
    InferRequest m_infer_request;

    Blob::Ptr m_input_blob;
    Blob::Ptr m_output_blob;
};

#endif //DEEPLAB_H
