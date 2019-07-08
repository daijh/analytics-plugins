// Copyright (C) <2019> Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>

#include <opencv2/opencv.hpp>

#include "common.h"
#include "deeplab.h"

using namespace std;
using namespace InferenceEngine;

Deeplab::Deeplab(std::string device) {
    dout << "Cons device: "<< device << endl;

    m_device = "CPU";
    m_model = "deeplabv3_mnv2-FP32/frozen_inference_graph.xml";
}

Deeplab::~Deeplab() {
    dout << endl;
}

bool Deeplab::init() {
    loadModel(m_device, m_model, &m_plugin, &m_infer_request, &m_input_blob, &m_output_blob);
    return true;
}

bool Deeplab::infer(cv::Mat& image, cv::Mat *segmentation)
{
    int image_width = image.cols;
    int image_height = image.rows;

    matU8ToBlob<uint8_t>(image, m_input_blob);

    dout << "+++do_infer" << endl;
    m_infer_request.Infer();
    dout << "---do_infer" << endl;

    const float *detections = m_output_blob->buffer().as<float *>();

    SizeVector output_dims = m_output_blob->getTensorDesc().getDims();
    cv::Mat orig_segmentation(output_dims[1], output_dims[2], CV_8U);

    for (size_t h = 0; h < output_dims[1]; h++) {
        for (size_t w = 0; w < output_dims[2]; w++) {
            unsigned char val = detections[h * output_dims[2] + w];

            //15 is person
            if (val == 15) {
                orig_segmentation.at<uchar>(h, w) = true;
            } else {
                orig_segmentation.at<uchar>(h, w) = false;
            }
        }
    }

    *segmentation = cv::Mat(image.size(), CV_8U);
    cv::resize(orig_segmentation, *segmentation, image.size());

    return true;
}

