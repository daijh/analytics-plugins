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
    , m_lr_height(0)
    , m_ratio(0.0) {
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

    cv::Mat resized_yuv_image(buffer->height + buffer->height / 2, buffer->width, CV_8UC1, buffer->buffer);

    if (m_lr_width != buffer->width || m_lr_height!= buffer->height) {
        dout << "Resize input image: "
            << buffer->width << "x" << buffer->height
            << " -> "
            << m_lr_width << "x" << m_lr_height
            << endl;

        resized_yuv_image.create(m_lr_height * 3 / 2, m_lr_width, CV_8UC1);

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
    }

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

rvaStatus FSRCNNPlugin::SetPluginParams(std::unordered_map<std::string, std::string> params) {
    if (!m_ready) {
        dout << "Not ready!" << endl;
    }

    std::unordered_map<std::string, std::string>::const_iterator got = params.find("ratio");
    if ( got == params.end() ) {
        dout << "Invalid to set plugin params" << endl;
        return RVA_ERR_OK;
    }

    m_ratio = atof(got->second.c_str());
    dout << "Set ratio " << got->second << " -> " << m_ratio << endl;

    m_sr_infer->enable(m_ratio > 0.0);

    return RVA_ERR_OK;
}

// Declare the plugin
DECLARE_PLUGIN(FSRCNNPlugin)

