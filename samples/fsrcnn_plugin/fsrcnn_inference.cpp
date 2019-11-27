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
    , m_enable(false)
    , m_frame_count(0) {
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

void FSRCNNInference::calcAlphaBeta(cv::Mat& img) {
    dout << "+++calc alpha/beta" <<  endl;

    // caculate alpha beta
    double min_gray = 0, max_gray = 0;
    int hist_size = 256;
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::minMaxLoc(gray, &min_gray, &max_gray);

    // current range
    float input_range = max_gray - min_gray;

    m_alpha = (hist_size) / input_range;
    m_beta = -min_gray * m_alpha;

    dout << "alpha:" << m_alpha << endl;
    dout << "beta:" << m_beta << endl;

    dout << "---calc alpha/beta" << endl;
    return;
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
        dout << "+++do SR" << endl;

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

        // convert I420 to BGR
        cv::Mat sr_img_bgr = cv::Mat(m_output_height, m_output_width, CV_8UC3);
        cv::cvtColor(sr_image, sr_img_bgr, cv::COLOR_YUV2BGR_I420);
        cv::Mat ROI = sr_img_bgr(cv::Rect(m_output_width/2, 0, m_output_width/2, m_output_height));
        cv::Mat ROI_cp = ROI.clone();

        // calc alpha/beta every 30 frames.
        if (!(m_frame_count % 30))
            calcAlphaBeta(ROI);
        m_frame_count++;
        dout << "---GammaCorrection" << endl;
        ROI.convertTo(ROI_cp, -1, m_alpha, m_beta);
        dout << "---GammaCorrection" << endl;

        // sharpen
        dout << "+++sharpen" << endl;
        cv::Mat sharpen_kernel = (cv::Mat_<char>(3, 3) << -1, -1, -1, -1, 9, -1, -1, -1, -1);
        filter2D(ROI_cp, ROI, -1, sharpen_kernel, cv::Point(-1, -1), 0, 0);
        // convert BGR to I420
        cv::cvtColor(sr_img_bgr, sr_image, cv::COLOR_BGR2YUV_I420);
        dout << "---sharpen" << endl;

        for (size_t i = 0; i < h; i++) {
            for (size_t j = w / 2 - 5; j < w / 2 + 5; j++) {
                sr_image.at<uchar>(i * w + j) = 145;
                sr_image.at<uchar>(w * h + i / 2 * w / 2 + j / 2) = 54;
                sr_image.at<uchar>(w * h * 5 / 4 + i /2 * w / 2 + j / 2) = 34;
            }
        }

        dout << "---do SR" << endl;
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

