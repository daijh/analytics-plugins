// Copyright (C) <2019> Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#ifndef FACE_REIDENTIFICATION_H
#define FACE_REIDENTIFICATION_H

#include <inference_engine.hpp>

using namespace InferenceEngine;

class FSRCNNInference {
public:
    FSRCNNInference(std::string device = "CPU");
    virtual ~FSRCNNInference();

    bool init();

    void calcAlphaBeta(cv::Mat& img);

    uint32_t getInputWidth() {return m_input_width;}
    uint32_t getInputHeight() {return m_input_height;}

    uint32_t getOutputWidth() {return m_output_width;}
    uint32_t getOutputHeight() {return m_output_height;}

    bool infer(cv::Mat& image, cv::Mat& sr_image);

    void enable(bool enable) {m_enable = enable;};

protected:
    void uploadToBlob(const cv::Mat& image, InferenceEngine::Blob::Ptr& blob);

private:
    std::string m_device;
    std::string m_model;

    InferencePlugin m_plugin;
    InferRequest m_infer_request;

    Blob::Ptr m_input_blob;
    Blob::Ptr m_output_blob;

    uint32_t m_input_width;
    uint32_t m_input_height;

    uint32_t m_output_width;
    uint32_t m_output_height;

    bool    m_enable;

    uint32_t m_frame_count;
    float m_alpha;
    float m_beta;
};

#endif //FACE_REIDENTIFICATION_H
