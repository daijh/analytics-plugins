// Copyright (C) <2019> Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#ifndef FACE_REIDENTIFICATION_H
#define FACE_REIDENTIFICATION_H

#include <inference_engine.hpp>

using namespace InferenceEngine;

class SuperResolutionInference {
public:
    SuperResolutionInference(std::string device = "CPU");
    virtual ~SuperResolutionInference();

    bool init();

    uint32_t getInputWidth() {return m_input_width;}
    uint32_t getInputHeight() {return m_input_height;}

    bool infer(cv::Mat& image, cv::Mat& sr_image);

private:
    std::string m_device;
    std::string m_model;

    InferencePlugin m_plugin;
    InferRequest m_infer_request;

    Blob::Ptr m_input_blob;
    Blob::Ptr m_output_blob;

    Blob::Ptr m_bic_input_blob;

    uint32_t m_input_width;
    uint32_t m_input_height;

    uint32_t m_output_width;
    uint32_t m_output_height;
};

#endif //FACE_REIDENTIFICATION_H
