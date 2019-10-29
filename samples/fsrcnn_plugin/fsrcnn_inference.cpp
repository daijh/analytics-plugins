// Copyright (C) <2019> Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>

#include "common.h"
#include "fsrcnn_inference.h"

using namespace std;
using namespace InferenceEngine;

FSRCNNInference::FSRCNNInference(std::string device)
    : m_input_width(0)
    , m_input_height(0)
    , m_output_width(0)
    , m_output_height(0) {
    dout << "Cons device: "<< device << endl;

    if (!device.compare("CPU")) {
        m_device = "CPU";
        m_model = "fsrcnn-model/fsrcnn_scale4.xml";
    } else if (!device.compare("GPU")) {
        m_device = "GPU";
        m_model = "fsrcnn-model/fsrcnn_scale4.xml";
    } else {
        m_device = "CPU";
        m_model = "fsrcnn-model/fsrcnn_scale4.xml";
    }
}

FSRCNNInference::~FSRCNNInference() {
    dout << endl;
}

bool FSRCNNInference::init() {
    loadModel(m_device, m_model, &m_plugin, &m_infer_request, &m_input_blob, &m_output_blob, Precision::FP32);

    m_input_width = m_input_blob->getTensorDesc().getDims()[3];
    m_input_height = m_input_blob->getTensorDesc().getDims()[2];

    m_output_width = m_output_blob->getTensorDesc().getDims()[3];
    m_output_height = m_output_blob->getTensorDesc().getDims()[2];

    return true;
}

bool FSRCNNInference::infer(cv::Mat& image, cv::Mat& sr_image)
{
    if (image.type() != CV_8UC1
        || image.size().width != m_input_width
        || image.size().height != m_input_height * 3 / 2) {

        dout << "Invalid input image type or resolution" << endl;
        return false;
    }

    uploadToBlob(image, m_input_blob);

    dout << "+++do_infer" << endl;
    m_infer_request.Infer();
    dout << "---do_infer" << endl;

    auto outputData = m_output_blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
    size_t h = m_output_blob->getTensorDesc().getDims()[2];
    size_t w = m_output_blob->getTensorDesc().getDims()[3];

    cv::Mat cubic_yuv_image(h, w, CV_8UC3);

    // fix me
    {
        cv::Mat orig_bgr_image;
        cv::cvtColor(image, orig_bgr_image, cv::COLOR_YUV2BGR_I420);

        cv::Mat resized_bgr_image;
        cv::resize(orig_bgr_image, resized_bgr_image, cv::Size(m_output_width, m_output_height), 0, 0, cv::INTER_CUBIC);

        cv::cvtColor(resized_bgr_image, cubic_yuv_image, cv::COLOR_BGR2YUV_I420);
    }

    sr_image.create(m_output_height * 3 / 2, m_output_width, CV_8UC1);

    for (size_t i = 0; i < h; i++) {
        for (size_t j = 0; j < w; j++) {
            float data;

            if (outputData[i * w + j] > 1.0)
                data = 1.0;
            else if (outputData[i * w + j] < 0.0)
                data = 0.0;
            else
                data = outputData[i * w + j];

            sr_image.at<uchar>(i * w + j) = data * 255.0;
        }
    }

    memcpy(sr_image.data, cubic_yuv_image.data, m_output_width * m_output_height / 2);

    return true;
}

void FSRCNNInference::uploadToBlob(const cv::Mat& image, InferenceEngine::Blob::Ptr& blob) {
    InferenceEngine::SizeVector blobSize = blob->getTensorDesc().getDims();
    const size_t width = blobSize[3];
    const size_t height = blobSize[2];
    float* blob_data = blob->buffer().as<float_t*>();

    for (size_t  h = 0; h < height; h++) {
        for (size_t w = 0; w < width; w++) {
            blob_data[h * width + w] = image.at<uchar>(h * width + w) / 255.0;
        }
    }

#if 1
    for(size_t i = 0; i < 10; i++)
        printf("%f\t", blob_data[i]);
    printf("\n");
#endif
}

