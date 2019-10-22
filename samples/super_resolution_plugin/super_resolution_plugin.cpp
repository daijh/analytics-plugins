// Copyright (C) <2019> Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <thread>

#include <opencv2/opencv.hpp>

#include "common.h"
#include "super_resolution_plugin.h"

using namespace std;

void init_worker(SuperResolutionPlugin *plugin) {
    plugin->init();
}

void SuperResolutionPlugin::init() {
    dout << endl;

    m_sr_infer = std::make_shared<SuperResolutionInference>("CPU");
    m_sr_infer->init();

    m_lr_width = m_sr_infer->getInputWidth();
    m_lr_height = m_sr_infer->getInputHeight();

    m_ready = true;
}

SuperResolutionPlugin::SuperResolutionPlugin()
    : m_ready(false)
    , m_init_thread(NULL)
    , m_lr_width(0)
    , m_lr_height(0) {
    dout << endl;

    m_init_thread = new std::thread(init_worker, this);
}

rvaStatus SuperResolutionPlugin::PluginInit(std::unordered_map<std::string, std::string> params) {
    dout << endl;
    return RVA_ERR_OK;
}

rvaStatus SuperResolutionPlugin::PluginClose() {
    dout << endl;
    return RVA_ERR_OK;
}

rvaStatus SuperResolutionPlugin::ProcessFrameAsync(std::unique_ptr<owt::analytics::AnalyticsBuffer> buffer) {
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
            m_frame_callback->OnPluginFrame(std::move(buffer));
        }

        return RVA_ERR_OK;
    }

    dout << "+++" << endl;

    cv::Mat yuv_image(buffer->height + buffer->height / 2, buffer->width, CV_8UC1, buffer->buffer);
    cv::Mat bgr_image(buffer->height, buffer->width, CV_8UC3);
    cv::cvtColor(yuv_image, bgr_image, cv::COLOR_YUV2BGR_I420);

    cv::Mat resized_bgr_image(bgr_image);
    if (m_lr_width != bgr_image.size().width || m_lr_height!= bgr_image.size().height) {
        cv::resize(bgr_image, resized_bgr_image, cv::Size(m_lr_width, m_lr_height));
    }

    cv::Mat sr_bgr_image;

    int ret = m_sr_infer->infer(resized_bgr_image, sr_bgr_image);
    if (!ret) {
        dout << "Infer error!" << endl;

        if (m_frame_callback) {
            m_frame_callback->OnPluginFrame(std::move(buffer));
        }

        return RVA_ERR_OK;
    }

    uint8_t *sr_yuv_data = new uint8_t[sr_bgr_image.size().width * sr_bgr_image.size().height * 3 / 2];
    cv::Mat sr_yuv_image(sr_bgr_image.size().height * 3 / 2, sr_bgr_image.size().width, CV_8UC1, sr_yuv_data);
    cv::cvtColor(sr_bgr_image, sr_yuv_image, cv::COLOR_BGR2YUV_I420);

    std::unique_ptr<owt::analytics::AnalyticsBuffer> sr_buffer(new owt::analytics::AnalyticsBuffer());
    sr_buffer->buffer = sr_yuv_data;
    sr_buffer->width = sr_bgr_image.size().width;
    sr_buffer->height = sr_bgr_image.size().height;

    if (m_frame_callback) {
        m_frame_callback->OnPluginFrame(std::move(sr_buffer));
    }

    dout << "---" << endl;

    return RVA_ERR_OK;
}

// Declare the plugin
DECLARE_PLUGIN(SuperResolutionPlugin)

