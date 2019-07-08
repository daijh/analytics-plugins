// Copyright (C) <2019> Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>

#include "common.h"
#include "face_reidentification.h"

using namespace std;
using namespace InferenceEngine;

static float ComputeReidDistance(const cv::Mat& descr1, const cv::Mat& descr2) {
#if 0
    printf("ComputeReidDistance, %dx%d, %dx%d\n",
            descr1.size().width, descr1.size().height,
            descr2.size().width, descr2.size().height
          );
#endif

    float xy = descr1.dot(descr2);
    float xx = descr1.dot(descr1);
    float yy = descr2.dot(descr2);
    float norm = sqrt(xx * yy) + 1e-6;

    return 1.0f - xy / norm;
}

FaceReidentification::FaceReidentification(std::string device) {
    dout << "Cons device: "<< device << endl;

    if (!device.compare("CPU")) {
        m_device = "CPU";
        m_model = "/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml";
    } else if (!device.compare("GPU")) {
        m_device = "GPU";
        m_model = "/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml";
    } else if (!device.compare("HDDL")) {
        m_device = "HDDL";
        m_model = "/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-reidentification-retail-0095/FP16/face-reidentification-retail-0095.xml";
    } else {
        m_device = "CPU";
        m_model = "/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml";
    }

    m_database = "/home/webrtc/face_features_database.txt";
    m_dist_threshold = 0.5;
}

FaceReidentification::~FaceReidentification() {
    dout << endl;
}

void FaceReidentification::loadFaceInfoDatabase(std::string &database) {
    std::ifstream features_file(database);
    std::string line;

    while (true) {
        FaceFeatureInfo info;

        std::getline(features_file, line);
        if (line.empty())
            break;
        info.name = line;

        std::getline(features_file, line);
        if (line.empty())
            break;
        int feature_num = atoi(line.c_str());

        info.feature_vec = cv::Mat(static_cast<int>(feature_num), 1, CV_32F);
        for(int i = 0; i < feature_num; i++) {
            std::getline(features_file, line);
            if (line.empty())
                break;

            info.feature_vec.at<float>(i) = strtof(line.c_str(),0);
        }

        if (line.empty())
            break;

        m_feature_database.push_back(info);

        cout << "load " << info.name << ", feature_num " << info.feature_vec.size().height << endl;
        cout << "\t";
        for(int i = 0; i < 16; i++) {
            cout << info.feature_vec.at<float>(i) << " ";
        }
        cout << endl;
    }

    return;
}

bool FaceReidentification::init() {
    loadFaceInfoDatabase(m_database);
    loadModel(m_device, m_model, &m_plugin, &m_infer_request, &m_input_blob, &m_output_blob);
    return true;
}

bool FaceReidentification::infer(cv::Mat& image, FaceReidentification::DetectedObject *result)
{
    cv::Mat feature_vec;

    infer(image, feature_vec);

    float dist = 100.0f;
    float min_dist = 100.0f;
    std::string name;

    for (auto &info : m_feature_database) {
        dist = ComputeReidDistance(feature_vec, info.feature_vec);
        if (dist < min_dist){
            min_dist = dist;
            name = info.name;
        }
    }

    std::cout << "Best match: " << name <<
        " , dist: " << std::fixed << dist <<
        " , dist_threshold: " << m_dist_threshold << endl;

    if (min_dist > m_dist_threshold)
        name = "Unknown";

    result->name = name;
    result->dist = min_dist;

    return true;
}

bool FaceReidentification::infer(cv::Mat& image, cv::Mat &feature_vec)
{
    int image_width = image.cols;
    int image_height = image.rows;

    matU8ToBlob<uint8_t>(image, m_input_blob);

    dout << "+++do_infer" << endl;
    m_infer_request.Infer();
    dout << "---do_infer" << endl;

    const float *detections = m_output_blob->buffer().as<float *>();
    SizeVector output_dims = m_output_blob->getTensorDesc().getDims();
    int feature_num = output_dims[1];

    feature_vec = cv::Mat(static_cast<int>(feature_num), 1, CV_32F);
    for(int i = 0; i < feature_num; i++) {
        feature_vec.at<float>(i) = detections[i];
    }

    for(int i = 0; i < 16; i++) {
        cout << feature_vec.at<float>(i) << " ";
    }
    cout << endl;

    return true;
}

