// Copyright (C) <2019> Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <thread>

#include <opencv2/opencv.hpp>

#include "common.h"
#include "face_replacement_plugin.h"

using namespace std;

void init_worker(FaceReplacementPlugin *plugin) {
    plugin->init();
}

void FaceReplacementPlugin::init() {
    m_avatar_file = "/home/webrtc/touxiang2.png";
    m_avatar_image = cv::imread(m_avatar_file.c_str(), cv::IMREAD_COLOR);
    if (m_avatar_image.empty()) {
        dout << "Read input image error: " << m_avatar_file << endl;
    } else {
        dout << "Read input image ok: " << m_avatar_file << endl;
    }

    m_detection.init();
    m_reidentification.init();

    m_ready = true;
}

FaceReplacementPlugin::FaceReplacementPlugin()
    : m_ready(false)
    , m_init_thread(NULL) {
    dout << endl;

    m_init_thread = new std::thread(init_worker, this);
}

rvaStatus FaceReplacementPlugin::PluginInit(std::unordered_map<std::string, std::string> params) {
    dout << endl;

    return RVA_ERR_OK;
}

rvaStatus FaceReplacementPlugin::PluginClose() {
    dout << endl;
    return RVA_ERR_OK;
}

rvaStatus FaceReplacementPlugin::ProcessFrameAsync(std::unique_ptr<owt::analytics::AnalyticsBuffer> buffer) {
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

    std::vector<FaceDetection::DetectedObject> result;
    m_detection.infer(bgr_image, result);

    for (auto &obj : result) {
        if (!m_avatar_image.empty()) {
            cv::Mat matRoi;
            matRoi = bgr_image(obj.rect);

            FaceReidentification::DetectedObject reid_result;
            m_reidentification.infer(matRoi, &reid_result);

            cv::Mat resized_avatar(obj.rect.height, obj.rect.width, CV_8UC3);
            cv::resize(m_avatar_image, resized_avatar, cv::Size(obj.rect.width, obj.rect.height));
            resized_avatar.copyTo(matRoi);

            char msg[128];
            snprintf(msg, 128, "%s: %.2f", reid_result.name.c_str(), reid_result.dist);
            //snprintf(msg, 128, "%s", reid_result.name.c_str());

            int font_face = cv::FONT_HERSHEY_COMPLEX_SMALL;
            double font_scale = 1;
            int thickness = 1;
            int baseline;
            cv::Size text_size = cv::getTextSize(msg, font_face, font_scale, thickness, &baseline);

            cv::Point origin;
            origin.y = obj.rect.y + text_size.height + 5;
            if (obj.rect.width > text_size.width)
                origin.x = obj.rect.x + (obj.rect.width - text_size.width) / 2;
            else
                origin.x = obj.rect.x;

            cv::putText(bgr_image,
                    msg,
                    origin,
                    font_face,
                    font_scale,
                    cv::Scalar(0, 0, 0),
                    thickness
                    );
        } else {
            cv::rectangle(bgr_image, obj.rect, cv::Scalar(255, 255, 0));

            char msg[128];
            snprintf(msg, 128, "%.2f", obj.confidence);

            int font_face = cv::FONT_HERSHEY_COMPLEX_SMALL;
            double font_scale = 1;
            int thickness = 1;
            int baseline;
            cv::Size text_size = cv::getTextSize(msg, font_face, font_scale, thickness, &baseline);

            cv::Point origin;
            origin.y = obj.rect.y;
            if (obj.rect.width > text_size.width)
                origin.x = obj.rect.x + (obj.rect.width - text_size.width) / 2;
            else
                origin.x = obj.rect.x;

            cv::putText(bgr_image,
                    msg,
                    origin,
                    font_face,
                    font_scale,
                    cv::Scalar(255, 255, 0),
                    thickness
                    );
        }
    }

    cv::cvtColor(bgr_image, yuv_image, cv::COLOR_BGR2YUV_I420);
    buffer->buffer = yuv_image.data;

    if (m_frame_callback) {
        m_frame_callback->OnPluginFrame(std::move(buffer));
    }

    dout << "---" << endl;

    return RVA_ERR_OK;
}

// Declare the plugin
DECLARE_PLUGIN(FaceReplacementPlugin)

