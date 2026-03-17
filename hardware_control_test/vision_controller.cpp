#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// --- CONFIGURATION DE LA CAMERA ---
const int CAMERA_WIDTH = 640;
const int CAMERA_HEIGHT = 480;

Mat regionOfInterest(Mat img) {
    Mat mask = Mat::zeros(img.size(), img.type());
    Point pts[4] = {
        Point(img.cols * 0.1, img.rows),
        Point(img.cols * 0.4, img.rows * 0.5),
        Point(img.cols * 0.6, img.rows * 0.5),
        Point(img.cols * 0.9, img.rows)
    };
    fillConvexPoly(mask, pts, 4, Scalar(255, 255, 255));
    Mat masked_image;
    bitwise_and(img, mask, masked_image);
    return masked_image;
}

vector<int> getAverageLine(Mat img, vector<Vec4i> lines, bool isLeft) {
    if (lines.empty()) return {0,0,0,0};

    vector<float> slopes, intercepts;
    for (size_t i = 0; i < lines.size(); i++) {
        int x1 = lines[i][0], y1 = lines[i][1], x2 = lines[i][2], y2 = lines[i][3];
        if (x1 == x2) continue; 
        float slope = (float)(y2 - y1) / (float)(x2 - x1);
        float intercept = y1 - slope * x1;
        if (isLeft && slope < -0.3) { slopes.push_back(slope); intercepts.push_back(intercept); }
        else if (!isLeft && slope > 0.3) { slopes.push_back(slope); intercepts.push_back(intercept); }
    }
    
    if (slopes.empty()) return {0,0,0,0};
    
    float avg_slope = 0, avg_intercept = 0;
    for (float s : slopes) avg_slope += s;
    for (float i : intercepts) avg_intercept += i;
    avg_slope /= slopes.size();
    avg_intercept /= intercepts.size();
    
    int y1 = img.rows, y2 = int(y1 * 0.6);
    int x1 = int((y1 - avg_intercept) / avg_slope);
    int x2 = int((y2 - avg_intercept) / avg_slope);
    
    return {x1, y1, x2, y2};
}

int main() {
    VideoCapture cap(0); 
    if (!cap.isOpened()) {
        cerr << "Erreur : Impossible d'ouvrir la caméra." << endl;
        return -1;
    }

    cap.set(CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH);
    cap.set(CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT);

    Mat frame, gray, blurImg, edges, roi;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, blurImg, Size(5, 5), 0);
        Canny(blurImg, edges, 50, 150);
        roi = regionOfInterest(edges);

        vector<Vec4i> lines;
        HoughLinesP(roi, lines, 2, CV_PI/180, 50, 40, 5);

        vector<int> leftLine = getAverageLine(frame, lines, true);
        vector<int> rightLine = getAverageLine(frame, lines, false);

        Mat display_img = frame.clone();
        int center_x = display_img.cols / 2;
        int target_x = center_x;

        if (leftLine[0] != 0) line(display_img, Point(leftLine[0], leftLine[1]), Point(leftLine[2], leftLine[3]), Scalar(255, 0, 0), 5);
        if (rightLine[0] != 0) line(display_img, Point(rightLine[0], rightLine[1]), Point(rightLine[2], rightLine[3]), Scalar(0, 0, 255), 5);

        if (leftLine[0] != 0 && rightLine[0] != 0) target_x = (leftLine[2] + rightLine[2]) / 2;
        else if (leftLine[0] != 0) target_x = leftLine[2] + (display_img.cols / 4); 
        else if (rightLine[0] != 0) target_x = rightLine[2] - (display_img.cols / 4);

        circle(display_img, Point(target_x, display_img.rows * 0.6), 8, Scalar(0, 255, 0), -1);
        line(display_img, Point(center_x, display_img.rows), Point(target_x, display_img.rows * 0.6), Scalar(0, 255, 0), 4);

        int error = target_x - center_x; 
        float kp = 0.5; 
        int steering_angle = error * kp;
        
        if (steering_angle > 30) steering_angle = 30;
        else if (steering_angle < -30) steering_angle = -30;

        int throttle = 100 - abs(steering_angle); 

        string control_text = "Angle: " + to_string(steering_angle) + " | Speed: " + to_string(throttle) + "%";
        putText(display_img, control_text, Point(20, 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

        imshow("Vision Controller (OpenCV)", display_img);

        if (waitKey(1) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
