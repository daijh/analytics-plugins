// Copyright (C) <2019> Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <thread>

#include <opencv2/opencv.hpp>

#include "common.h"
#include "fsrcnn_plugin.h"

#include "libyuv/scale.h"

using namespace std;

void init_worker(FSRCNNPlugin *plugin) {
    plugin->init();
}

void FSRCNNPlugin::init() {
    dout << endl;

    m_sr_infer = std::make_shared<FSRCNNInference>("CPU");
    m_sr_infer->init();

    m_lr_width = m_sr_infer->getInputWidth();
    m_lr_height = m_sr_infer->getInputHeight();

    m_sr_width = m_sr_infer->getOutputWidth();
    m_sr_height = m_sr_infer->getOutputHeight();

    m_ready = true;
}

FSRCNNPlugin::FSRCNNPlugin()
    : m_ready(false)
    , m_init_thread(NULL)
    , m_lr_width(0)
    , m_lr_height(0) {
    dout << endl;

    m_init_thread = new std::thread(init_worker, this);
}

rvaStatus FSRCNNPlugin::PluginInit(std::unordered_map<std::string, std::string> params) {
    dout << endl;
    return RVA_ERR_OK;
}

rvaStatus FSRCNNPlugin::PluginClose() {
    dout << endl;
    return RVA_ERR_OK;
}

rvaStatus FSRCNNPlugin::ProcessFrameAsync(std::unique_ptr<owt::analytics::AnalyticsBuffer> buffer) {
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

#define LIBYUV

#ifdef LIBYUV
    cv::Mat resized_yuv_image(m_lr_height * 3 / 2, m_lr_width, CV_8UC1);

    libyuv::I420Scale(
            buffer->buffer,
            buffer->width,
            buffer->buffer + buffer->width * buffer->height,
            buffer->width / 2,
            buffer->buffer + buffer->width * buffer->height * 5 / 4,
            buffer->width / 2,
            buffer->width, buffer->height,
            resized_yuv_image.data,
            m_lr_width,
            resized_yuv_image.data + m_lr_width * m_lr_height,
            m_lr_width / 2,
            resized_yuv_image.data + m_lr_width * m_lr_height * 5 /4,
            m_lr_width / 2,
            m_lr_width, m_lr_height,
            libyuv::kFilterBox); //kFilterBilinear

    dout << "libyuv" << endl;
#else
    cv::Mat yuv_image(buffer->height + buffer->height / 2, buffer->width, CV_8UC1, buffer->buffer);
    cv::Mat bgr_image(buffer->height, buffer->width, CV_8UC3);
    cv::cvtColor(yuv_image, bgr_image, cv::COLOR_YUV2BGR_I420);

    cv::Mat resized_bgr_image(bgr_image);
    if (m_lr_width != bgr_image.size().width || m_lr_height!= bgr_image.size().height) {
        dout << "Resize input image: "
            << bgr_image.size().width << "x" << bgr_image.size().height
            << " -> "
            << m_lr_width << "x" << m_lr_height
            << endl;

        cv::resize(bgr_image, resized_bgr_image, cv::Size(m_lr_width, m_lr_height));
    }
    cv::Mat resized_yuv_image;
    cv::cvtColor(resized_bgr_image, resized_yuv_image, cv::COLOR_BGR2YUV_I420);

    dout << "opencv" << endl;
#endif

    uint8_t *sr_yuv_data = new uint8_t[m_sr_width * m_sr_height * 3 / 2];
    cv::Mat sr_yuv_image(m_sr_height * 3 / 2, m_sr_width, CV_8UC1, sr_yuv_data);

    int ret = m_sr_infer->infer(resized_yuv_image, sr_yuv_image);
    if (!ret) {
        dout << "Infer error!" << endl;

        if (m_frame_callback) {
            m_frame_callback->OnPluginFrame(std::move(buffer));
        }

        return RVA_ERR_OK;
    }

    std::unique_ptr<owt::analytics::AnalyticsBuffer> sr_buffer(new owt::analytics::AnalyticsBuffer());
    sr_buffer->buffer = sr_yuv_data;
    sr_buffer->width = m_sr_width;
    sr_buffer->height = m_sr_height;

    if (m_frame_callback) {
        m_frame_callback->OnPluginFrame(std::move(sr_buffer));
    }

    dout << "---" << endl;

    return RVA_ERR_OK;
}

// Declare the plugin
DECLARE_PLUGIN(FSRCNNPlugin)

