// Copyright (C) <2019> Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#ifndef FACE_REPLACEMENT_PLUGIN_H
#define FACE_REPLACEMENT_PLUGIN_H

#include <memory>

#include "plugin.h"
#include "fsrcnn_inference.h"

class FSRCNNPlugin : public rvaPlugin {
public:
    FSRCNNPlugin();

    virtual rvaStatus PluginInit(std::unordered_map<std::string, std::string> params);

    virtual rvaStatus PluginClose();

    virtual rvaStatus GetPluginParams(std::unordered_map<std::string, std::string>& params) {
        return RVA_ERR_OK;
    }

    virtual rvaStatus SetPluginParams(std::unordered_map<std::string, std::string> params);

    virtual rvaStatus ProcessFrameAsync(std::unique_ptr<owt::analytics::AnalyticsBuffer> buffer);

    virtual rvaStatus RegisterFrameCallback(rvaFrameCallback* pCallback) {
        m_frame_callback = pCallback;
        return RVA_ERR_OK;
    }

    virtual rvaStatus DeRegisterFrameCallback() {
        m_frame_callback = nullptr;
        return RVA_ERR_OK;
    }

    virtual rvaStatus RegisterEventCallback(rvaEventCallback* pCallback) {
        m_event_callback = pCallback;
        return RVA_ERR_OK;
    }

    virtual rvaStatus DeRegisterEventCallback() {
        m_event_callback = nullptr;
        return RVA_ERR_OK;
    }

    void init();

private:
    rvaFrameCallback *m_frame_callback;
    rvaEventCallback *m_event_callback;

    bool m_ready;
    std::thread *m_init_thread;

    std::shared_ptr<FSRCNNInference> m_sr_infer;
    uint32_t m_lr_width;
    uint32_t m_lr_height;
    uint32_t m_sr_width;
    uint32_t m_sr_height;

    float m_ratio;
};

#endif  //FACE_REPLACEMENT_PLUGIN_H
