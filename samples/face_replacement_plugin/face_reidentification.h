// Copyright (C) <2019> Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#ifndef FACE_REIDENTIFICATION_H
#define FACE_REIDENTIFICATION_H

#include <inference_engine.hpp>

using namespace InferenceEngine;

class FaceReidentification {
public:
    struct DetectedObject {
        std::string name;
        float dist;
    };

    struct FaceFeatureInfo {
        std::string name;
        cv::Mat feature_vec;
    };

public:
    FaceReidentification(std::string device = "CPU");
    virtual ~FaceReidentification();

    bool init();

    bool infer(cv::Mat& image, DetectedObject *result);
    bool infer(cv::Mat& image, cv::Mat &feature_vec);

protected:
    void loadFaceInfoDatabase(std::string &database);

private:
    std::string m_device;
    std::string m_model;
    std::string m_database;

    InferencePlugin m_plugin;
    InferRequest m_infer_request;

    Blob::Ptr m_input_blob;
    Blob::Ptr m_output_blob;

    std::vector<FaceFeatureInfo> m_feature_database;

    float m_dist_threshold;
};

#endif //FACE_REIDENTIFICATION_H
