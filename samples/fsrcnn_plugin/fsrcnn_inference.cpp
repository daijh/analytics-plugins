// Copyright (C) <2019> Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>

#include "common.h"
#include "fsrcnn_inference.h"

#include "libyuv/scale.h"

using namespace std;
using namespace InferenceEngine;

FSRCNNInference::FSRCNNInference(std::string device)
    : m_input_width(0)
    , m_input_height(0)
    , m_output_width(0)
    , m_output_height(0)
    , m_enable(false) {
    dout << "Cons device: "<< device << endl;

    if (!device.compare("CPU")) {
        m_device = "CPU";
        m_model = "fsrcnn-model/fsrcnn.xml";
    } else if (!device.compare("GPU")) {
        m_device = "GPU";
        m_model = "fsrcnn-model/fsrcnn.xml";
    } else {
        m_device = "CPU";
        m_model = "fsrcnn-model/fsrcnn.xml";
    }
}

FSRCNNInference::~FSRCNNInference() {
    dout << endl;
}

bool FSRCNNInference::init() {
    loadModel(m_device, m_model, &m_plugin, &m_infer_request, &m_input_blob, &m_output_blob, Precision::FP32);

    m_input_width = m_input_blob->getTensorDesc().getDims()[2];
    m_input_height = m_input_blob->getTensorDesc().getDims()[3];

    m_output_width = m_output_blob->getTensorDesc().getDims()[2];
    m_output_height = m_output_blob->getTensorDesc().getDims()[3];

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

    if (sr_image.type() != CV_8UC1
        || sr_image.size().width != m_output_width
        || sr_image.size().height != m_output_height * 3 / 2) {

        dout << "Invalid output image type or resolution" << endl;
        return false;
    }

    uploadToBlob(image, m_input_blob);

    dout << "+++do_infer" << endl;
    m_infer_request.Infer();
    dout << "---do_infer" << endl;

    auto outputData = m_output_blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
    size_t h = m_output_blob->getTensorDesc().getDims()[3];
    size_t w = m_output_blob->getTensorDesc().getDims()[2];

    uint8_t *cubic_yuv_data = sr_image.data;

    libyuv::I420Scale(
            image.data,
            m_input_width,
            image.data + m_input_width * m_input_height,
            m_input_width / 2,
            image.data + m_input_width * m_input_height * 5 / 4,
            m_input_width / 2,
            m_input_width, m_input_height,
            cubic_yuv_data,
            m_output_width,
            cubic_yuv_data + m_output_width * m_output_height,
            m_output_width / 2,
            cubic_yuv_data + m_output_width * m_output_height * 5 /4,
            m_output_width / 2,
            m_output_width, m_output_height,
            libyuv::kFilterBox); //kFilterBilinear

    if (m_enable) {
        dout << "do SR" << endl;

        for (size_t i = 0; i < h; i++) {
            for (size_t j = w / 2; j < w; j++) {
                float data;

                if (outputData[j * h + i] > 1.0)
                    data = 1.0;
                else if (outputData[j * h + i] < 0.0)
                    data = 0.0;
                else
                    data = outputData[j * h + i];

                sr_image.at<uchar>(i * w + j) = data * 255.0;
            }
        }
    }

    return true;
}

void FSRCNNInference::uploadToBlob(const cv::Mat& image, InferenceEngine::Blob::Ptr& blob) {
    InferenceEngine::SizeVector blobSize = blob->getTensorDesc().getDims();
    const size_t width = blobSize[2];
    const size_t height = blobSize[3];
    float* blob_data = blob->buffer().as<float_t*>();

    for (size_t  h = 0; h < height; h++) {
        for (size_t w = 0; w < width; w++) {
            blob_data[w * height+ h] = image.at<uchar>(h * width + w) / 255.0;
        }
    }
}

