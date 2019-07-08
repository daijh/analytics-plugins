//Copyright (C) <2019> Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <stdio.h>
#include <getopt.h>
#include <unistd.h>

#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>

#include "common.h"
#include "face_detection.h"
#include "face_reidentification.h"

using namespace std;

static void usage(int argc, char *argv[]) {
    fprintf(stderr, "usage: %s\n", argv[0]);
    fprintf(stderr, "   -i  input file path\n");
    fprintf(stderr, "   -h  help\n");

    fprintf(stderr, "example:\n");
    fprintf(stderr, "   %s  -i  someone.png\n", argv[0]);

    exit(1);
}

int main(int argc, char* argv[]) {
    string I_in_filename;

    // parse input parameters
    int c;

    while(1) {
        c = getopt_long(argc, argv, "i:h",
                NULL, NULL);
        if (c == -1)
            break;

        switch (c) {
            case 'i':
                I_in_filename = optarg;
                break;

            case 'h':
                usage(argc, argv);
                break;

            default:
                cout << "Invalid opt: " << c << endl;
                usage(argc, argv);
                break;
        }
    }

    if (I_in_filename.empty()) {
        cout << "Error, no input file" << endl;

        usage(argc, argv);
        return 1;
    }

    cv::Mat image = cv::imread(I_in_filename.c_str(), cv::IMREAD_COLOR);
    if (image.empty()) {
        cout << "Read input image error: " << I_in_filename << endl;
        return 1;
    }

    FaceDetection detection;
    detection.init();

    std::vector<FaceDetection::DetectedObject> result;
    detection.infer(image, result);

    if (result.size() != 1) {
        cout << "Face detection error" << endl;
        return 1;
    }

    FaceReidentification reidentification;
    reidentification.init();

    for (auto &obj : result) {
        cv::Mat matRoi = image(obj.rect);

        size_t pos = I_in_filename.rfind('.');
        string name = I_in_filename;
        name.insert(pos, "-feature");
        cv::imwrite(name.c_str(), matRoi);

        cout << "Write feature picture: " << name << endl;

        cv::Mat feature_vec;
        reidentification.infer(matRoi, feature_vec);

        {
            FaceReidentification::DetectedObject result;
            reidentification.infer(matRoi, &result);
        }

        pos = name.rfind('.');
        name.replace(pos, name.size() - pos, ".txt");
        cout << "Write feature txt: " << name << endl;

        ofstream feature_vec_output;
        feature_vec_output.open(name, std::ofstream::trunc);
        if (!feature_vec_output.is_open()) {
            cout<< "Unable to open file: " << name << endl;
            return 1;
        }

        size_t lpos = I_in_filename.rfind('/');
        size_t rpos = I_in_filename.rfind('.');
        name = I_in_filename.substr(lpos + 1, rpos - lpos - 1);

        cout << "Feature name: " << name << endl;
        feature_vec_output << name << endl;

        cout << "Feature number: " << feature_vec.size().height << endl;
        feature_vec_output << feature_vec.size().height << endl;
        for(int i = 0; i < feature_vec.size().height; i++) {
            feature_vec_output << feature_vec.at<float>(i) << endl;
        }
    }

    return 0;
}
