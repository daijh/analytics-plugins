//Copyright (C) <2019> Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <stdio.h>
#include <getopt.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>

#include "common.h"
#include "deeplab.h"

using namespace std;

static void usage(int argc, char *argv[]) {
    fprintf(stderr, "usage: %s\n", argv[0]);
    fprintf(stderr, "   -i  input file path\n");
    fprintf(stderr, "   -b  background image path\n");
    fprintf(stderr, "   -s  show output\n");
    fprintf(stderr, "   -h  help\n");

    fprintf(stderr, "example:\n");
    fprintf(stderr, "   %s  -i  someone.png\n", argv[0]);

    exit(1);
}

int main(int argc, char* argv[]) {
    string I_in_filename;
    bool I_show = false;
    std::string m_bg_file = "background.jpg";

    // parse input parameters
    int c;

    while(1) {
        c = getopt_long(argc, argv, "i:b:sh",
                NULL, NULL);
        if (c == -1)
            break;

        switch (c) {
            case 'i':
                I_in_filename = optarg;
                break;

            case 'b':
                m_bg_file = optarg;
                break;

            case 's':
                I_show = true;
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

    cv::Mat m_bg_image = cv::imread(m_bg_file.c_str(), cv::IMREAD_COLOR);
    if (m_bg_image.empty()) {
        dout << "Read input image error: " << m_bg_file << endl;
    } else {
        dout << "Read input image ok: " << m_bg_file << endl;
    }

    Deeplab *m_deeplab = new Deeplab("CPU");
    m_deeplab->init();

    cv::Mat segmentation;
    m_deeplab->infer(image, &segmentation);

    cv::Mat new_image(image.size(), CV_8UC3, cv::Scalar(255, 255, 0));
    if (!m_bg_image.empty())
        cv::resize(m_bg_image, new_image, new_image.size());

    for (size_t h = 0; h < image.size().height; h++) {
        for (size_t w = 0; w < image.size().width; w++) {
            if(segmentation.at<uchar>(h, w)) {
                   new_image.at<cv::Vec3b>(h, w)[0] = image.at<cv::Vec3b>(h, w)[0];
                   new_image.at<cv::Vec3b>(h, w)[1] = image.at<cv::Vec3b>(h, w)[1];
                   new_image.at<cv::Vec3b>(h, w)[2] = image.at<cv::Vec3b>(h, w)[2];
            }
        }
    }

    cv::Mat output(cv::Size(image.size().width * 2, image.size().height), CV_8UC3);

    cv::Mat matRoi;
    matRoi = output(cv::Rect(0, 0, image.size().width, image.size().height));
    image.copyTo(matRoi);
    matRoi = output(cv::Rect(image.size().width, 0, image.size().width, image.size().height));
    new_image.copyTo(matRoi);

    if (I_show) {
        cv::imshow("output", output);
        cv::waitKey(0);
    } else {
        char name[128];
        snprintf(name, 128, "output.jpg");
        cv::imwrite(name, output);
        printf("Dump %s\n", name);
    }

    return 0;
}
