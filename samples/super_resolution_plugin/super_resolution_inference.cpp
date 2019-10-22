// Copyright (C) <2019> Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>

#include "common.h"
#include "super_resolution_inference.h"

using namespace std;
using namespace InferenceEngine;

SuperResolutionInference::SuperResolutionInference(std::string device)
    : m_input_width(0)
    , m_input_height(0)
    , m_output_width(0)
    , m_output_height(0) {
    dout << "Cons device: "<< device << endl;

    if (!device.compare("CPU")) {
        m_device = "CPU";
        m_model = "/opt/intel/computer_vision_sdk/deployment_tools/intel_models/single-image-super-resolution-1011/FP32/single-image-super-resolution-1011.xml";
    } else if (!device.compare("GPU")) {
        m_device = "GPU";
        m_model = "/opt/intel/computer_vision_sdk/deployment_tools/intel_models/single-image-super-resolution-1011/FP32/single-image-super-resolution-1011.xml";
    } else if (!device.compare("HDDL")) {
        m_device = "HDDL";
        m_model = "/opt/intel/computer_vision_sdk/deployment_tools/intel_models/single-image-super-resolution-1011/FP16/single-image-super-resolution-1011.xml";
    } else {
        m_device = "CPU";
        m_model = "/opt/intel/computer_vision_sdk/deployment_tools/intel_models/single-image-super-resolution-1011/FP32/single-image-super-resolution-1011.xml";
    }
}

SuperResolutionInference::~SuperResolutionInference() {
    dout << endl;
}

bool SuperResolutionInference::init() {
    loadModel(m_device, m_model, &m_plugin, &m_infer_request, &m_input_blob, &m_output_blob, Precision::FP32);

    const std::string bicInputBlobName = "1";
    m_bic_input_blob = m_infer_request.GetBlob(bicInputBlobName);

    m_input_width = m_input_blob->getTensorDesc().getDims()[3];
    m_input_height = m_input_blob->getTensorDesc().getDims()[2];

    m_output_width = m_output_blob->getTensorDesc().getDims()[3];
    m_output_height = m_output_blob->getTensorDesc().getDims()[2];

    dout << "two inputs" << endl;

    return true;
}

bool SuperResolutionInference::infer(cv::Mat& image, cv::Mat& sr_image)
{
    if (image.size().width != m_input_width
        || image.size().height != m_input_height) {
        dout << "Invalid input image resolution " << image.size().width << "x" << image.size().height << endl;
        return false;
    }

    matU8ToBlob<float_t>(image, m_input_blob);

    cv::Mat up_image;
    cv::resize(image, up_image, cv::Size(m_output_width, m_output_height), 0, 0, cv::INTER_CUBIC);
    matU8ToBlob<float_t>(up_image, m_bic_input_blob);

    dout << "+++do_infer" << endl;
    m_infer_request.Infer();
    dout << "---do_infer" << endl;

    auto outputData = m_output_blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
    size_t h = m_output_blob->getTensorDesc().getDims()[2];
    size_t w = m_output_blob->getTensorDesc().getDims()[3];
    size_t nunOfPixels = w * h;

    std::vector<cv::Mat> imgPlanes{cv::Mat(h, w, CV_32FC1, &(outputData[0])),
        cv::Mat(h, w, CV_32FC1, &(outputData[nunOfPixels])),
        cv::Mat(h, w, CV_32FC1, &(outputData[nunOfPixels * 2]))};
    for (auto & img : imgPlanes)
        img.convertTo(img, CV_8UC1, 255);

    cv::merge(imgPlanes, sr_image);

#if 1
    {
        cv::Mat matRoi;
        cv::Mat srcMatRoi;

        //srcMatRoi = up_image(cv::Rect(0, 0, m_output_width / 2, m_output_height));
        srcMatRoi = up_image(cv::Rect(m_output_width / 2, 0, m_output_width / 2, m_output_height));
        matRoi = sr_image(cv::Rect(0, 0, m_output_width / 2, m_output_height));
        srcMatRoi.copyTo(matRoi);
    }
#endif

#if 0 //dump
    {
        cv::Mat result(cv::Size(m_output_width, m_output_height), CV_8UC3);
        cv::Mat matRoi;
        cv::Mat srcMatRoi;

        srcMatRoi = up_image(cv::Rect(0, 0, m_output_width / 2, m_output_height));
        matRoi = result(cv::Rect(0, 0, m_output_width / 2, m_output_height));
        srcMatRoi.copyTo(matRoi);

        srcMatRoi = sr_image(cv::Rect(0, 0, m_output_width / 2, m_output_height));
        matRoi = result(cv::Rect(m_output_width / 2, 0, m_output_width / 2, m_output_height));
        srcMatRoi.copyTo(matRoi);

        char name[128] = "mix.png";
        cv::imwrite(name, result);
        printf("Dump %s\n", name);
    }
#endif

    return true;
}

