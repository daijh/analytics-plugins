// Copyright (C) <2019> Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <thread>

#include <opencv2/opencv.hpp>

#include "common.h"
#include "bg_replacement_plugin.h"

using namespace std;

void init_worker(BgReplacementPlugin *plugin) {
    plugin->init();
}

void BgReplacementPlugin::init() {
    m_bg_file = "background.jpg";
    m_bg_image = cv::imread(m_bg_file.c_str(), cv::IMREAD_COLOR);
    if (m_bg_image.empty()) {
        dout << "Read input image error: " << m_bg_file << endl;
    } else {
        dout << "Read input image ok: " << m_bg_file << endl;
    }

    m_deeplab = new Deeplab("CPU");
    m_deeplab->init();

    m_ready = true;
}

BgReplacementPlugin::BgReplacementPlugin()
    : m_ready(false)
    , m_init_thread(NULL) {
    dout << endl;

    m_init_thread = new std::thread(init_worker, this);
}

rvaStatus BgReplacementPlugin::PluginInit(std::unordered_map<std::string, std::string> params) {
    dout << endl;
    return RVA_ERR_OK;
}

rvaStatus BgReplacementPlugin::PluginClose() {
    dout << endl;
    return RVA_ERR_OK;
}

rvaStatus BgReplacementPlugin::ProcessFrameAsync(std::unique_ptr<owt::analytics::AnalyticsBuffer> buffer) {
    if (!buffer || !buffer->buffer) {
        dout << "Invalid buffer!" << endl;

        if (m_frame_callback) {
            m_frame_callback->OnPluginFrame(std::move(buffer));
        }

        return RVA_ERR_OK;
    }

    if (!m_ready) {
        dout << "Not ready!" << endl;

        if (m_frame_callback) {
            if (!m_bg_image.empty()) {
                cv::Mat dummy_bgr_out(buffer->height, buffer->width, CV_8UC3, cv::Scalar(255, 255, 0));
                cv::resize(m_bg_image, dummy_bgr_out, dummy_bgr_out.size());

                cv::Mat dummy_yuv_image(buffer->height + buffer->height / 2, buffer->width, CV_8UC1, buffer->buffer);
                cv::cvtColor(dummy_bgr_out, dummy_yuv_image, cv::COLOR_BGR2YUV_I420);
                buffer->buffer = dummy_yuv_image.data;
            }

            m_frame_callback->OnPluginFrame(std::move(buffer));
        }

        return RVA_ERR_OK;
    }

    dout << "+++" << endl;

    cv::Mat yuv_image(buffer->height + buffer->height / 2, buffer->width, CV_8UC1, buffer->buffer);
    cv::Mat bgr_image(buffer->height, buffer->width, CV_8UC3);
    cv::cvtColor(yuv_image, bgr_image, cv::COLOR_YUV2BGR_I420);

    cv::Mat segmentation;
    m_deeplab->infer(bgr_image, &segmentation);

    cv::Mat bgr_image_out(buffer->height, buffer->width, CV_8UC3, cv::Scalar(255, 255, 0));
    if (!m_bg_image.empty())
        cv::resize(m_bg_image, bgr_image_out, bgr_image_out.size());

    for (size_t h = 0; h < bgr_image.size().height; h++) {
        for (size_t w = 0; w < bgr_image.size().width; w++) {
            if(segmentation.at<uchar>(h, w)) {
                   bgr_image_out.at<cv::Vec3b>(h, w)[0] = bgr_image.at<cv::Vec3b>(h, w)[0];
                   bgr_image_out.at<cv::Vec3b>(h, w)[1] = bgr_image.at<cv::Vec3b>(h, w)[1];
                   bgr_image_out.at<cv::Vec3b>(h, w)[2] = bgr_image.at<cv::Vec3b>(h, w)[2];
            }
        }
    }

    cv::cvtColor(bgr_image_out, yuv_image, cv::COLOR_BGR2YUV_I420);
    buffer->buffer = yuv_image.data;

    if (m_frame_callback) {
        m_frame_callback->OnPluginFrame(std::move(buffer));
    }

    dout << "---" << endl;

    return RVA_ERR_OK;
}

// Declare the plugin
DECLARE_PLUGIN(BgReplacementPlugin)

