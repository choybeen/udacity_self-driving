#include <stdio.h>
#include "stdafx.h"
#include <iostream>
#include "funcdef.h"

#include <iomanip>

#include "matinrange.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

Vec4f drawfittestline(Mat img, Vec4f line_para, int p1, int p2, bool bfx, Scalar color);
int lane_detection_udacity()
{
	bool bshowmid = true;
//	std::string image_path = samples::findFile("PyramidPattern.jpg");
//	Mat img = imread(image_path, IMREAD_COLOR);
	cv::String path("D:\\jupyterNote\\CarND-LaneLines-P1-master\\test_images/*.jpg"); //select only jpg
	vector<cv::String> fn;
	vector<cv::Mat> original_images, original_small;
	vector<cv::Mat> canny_segmented_images;
	cv::glob(path, fn, true); // recurse
	//step1,  read images and convert to hsv and hsl
	for (size_t k = 0; k < fn.size(); ++k)
	{
		if (k < 10 || k>10) continue;
		cv::Mat small, img = cv::imread(fn[k], IMREAD_COLOR);
		if (img.empty()) continue; //only proceed if sucsessful		
		resize(img, small, Size(img.cols/2, img.rows/2));
		original_images.emplace_back(img);// you probably want to do some preprocessing
		original_small.emplace_back(small);// you probably want to do some preprocessing
	}
	imshow("0", original_images[0]);
	waitKey(0);
	
    //step2, thread color image only keep yellow and white
//	cvinRange::threadcolor(original_images);
	Scalar yellowlow(15, 38, 115), yellowhigh(35, 204, 255);
	Scalar whitelow(0, 200, 0), whitehigh(170, 255, 255);
	vector<cv::Mat> hsv_images, hls_images;
	vector<cv::Mat> hls_yellow_images;
	vector<cv::Mat> hls_white_images;
	for (int i = 0; i < original_small.size(); i++) {
		Mat small = original_small[i];
		Mat hsv, hls;
		cvtColor(small, hsv, cv::COLOR_BGR2HSV);
	    cvtColor(small, hls, cv::COLOR_BGR2HLS);
		hsv_images.emplace_back(hsv);
		hls_images.emplace_back(hls);
		Mat hls_yellow, hls_white;
		inRange(hls, yellowlow, yellowhigh, hls_yellow);
		inRange(hls, whitelow, whitehigh, hls_white);
		hls_yellow_images.emplace_back(hls_yellow);
		hls_white_images.emplace_back(hls_white);
		if (bshowmid) {
			imshow("0", original_small[i]);
			imshow("1", hls_yellow);
			imshow("2", hls_white);
			waitKey(0);
		}
	}

    //step3, from yellow part to gray image, possible lane areas
	vector<cv::Mat> combined_hsl_images;
	vector<cv::Mat> grayscale_images;
	for (int i = 0; i < hls_white_images.size(); i++) {
		Mat hsl_mask, hsl_filtered, gray;
		bitwise_or(hls_yellow_images[i], hls_white_images[i], hsl_mask);
		//bitwise_and(original_images[i], hsl_mask, hsl_filtered);
		original_small[i].copyTo(hsl_filtered, hsl_mask);
		combined_hsl_images.emplace_back(hsl_filtered);
		cvtColor(hsl_filtered, gray, cv::COLOR_BGR2GRAY);
		grayscale_images.emplace_back(gray);
		if (bshowmid) {
			imshow("0", original_small[i]);
			imshow("1", hsl_filtered);
			imshow("2", gray);
			waitKey(0);
		}
	}
	//step4, blur image with filter to remove noises, and detect edge by canny
	vector<cv::Mat> canny_images;
	int lowThreshold = 0;
	const int max_lowThreshold = 90;
	const int ratio = 3;
	const int kernel_size = 3;
	for (int i = 0; i < hls_white_images.size(); i++) {
		Mat blurgaussian, detected_edges;
		cv::GaussianBlur(grayscale_images[i], blurgaussian, Size(5, 5), 0, 0); // blur(src_gray, detected_edges, Size(3, 3));
		cv::Canny(blurgaussian, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);
		canny_images.emplace_back(detected_edges);
		if (bshowmid) {
			imshow("0", original_small[i]);
			imshow("1", blurgaussian);
			imshow("2", detected_edges);
			waitKey(0);
		}
	}

	//step5, remove other areas by ROI, keep only front area before car
	for (int i = 0; i < hls_white_images.size(); i++) {
		Mat edges = canny_images[i];
		int w = edges.cols, h = edges.rows;
		Mat mask(h, w, CV_8U, cv::Scalar::all(255));
		vector< Point> contour;
		if (edges.rows == 540/2) {
			contour.push_back(Point(410/2, 330/2));
			contour.push_back(Point(650/2, 350/2));
			contour.push_back(Point(w - 30/2, h - 1));
			contour.push_back(Point(130/2, h - 1));
		}
		else {
			contour.push_back(Point(600/2, 450/2));
			contour.push_back(Point(750/2, 450/2));
			contour.push_back(Point(1160/2, h - 1));
			contour.push_back(Point(160/2, h - 1));
		}
		const Point *pts = (const cv::Point*) Mat(contour).data;
		int npts = Mat(contour).rows;		
//		polylines(mask, &pts, &npts, 1, true, cv::Scalar::all(255), 1);	// draw the polygon 
		fillPoly(mask, &pts, &npts, 1, cv::Scalar::all(0));	// draw the polygon 
		edges.setTo(Scalar::all(0), mask);
		canny_segmented_images.emplace_back(edges);
		if (bshowmid) {
			imshow("0", original_small[i]);
			imshow("1", mask);
			imshow("2", edges);
			waitKey(0);
		}
		stringstream ss;
		ss << "D:\\jupyterNote\\CarND-LaneLines-P1-master\\edge_images\\" << setw(2) << setfill('0') << i << ".jpg";
		cv::String path = ss.str();
		//imwrite(path, edges);
	}
/*
	cv::String path2("D:\\jupyterNote\\CarND-LaneLines-P1-master\\edge_images/*.jpg"); //select only jpg
	cv::glob(path2, fn, true); // recurse
	//step1,  read images and convert to hsv and hsl
	for (size_t k = 0; k < fn.size(); ++k)
	{
		cv::Mat img = cv::imread(fn[k], IMREAD_GRAYSCALE);
		if (img.empty()) continue; //only proceed if sucsessful		
		canny_segmented_images.emplace_back(img);// you probably want to do some preprocessing
	}
	*/
    //step6, apply the Hough Transform
	vector<vector<Vec4i>> alllines; // hold all lines
	for (int i = 0; i < canny_segmented_images.size(); i++) {
		Mat edge = canny_segmented_images[i];
		Mat cdst = original_small[i].clone();
		Mat cdstP = original_small[i].clone();
		int w = edge.cols, h = edge.rows;
		vector<Vec2f> lines; // will hold the results of the detection
		HoughLines(edge, lines, 1, CV_PI / 180, 50/2, 0, 0); // runs the actual detection
		// Draw the lines
		for (size_t i = 0; i < lines.size(); i++)
		{
			float rho = lines[i][0], theta = lines[i][1];
			Point pt1, pt2;
			double a = cos(theta), b = sin(theta);
			double x0 = a * rho, y0 = b * rho;
			pt1.x = cvRound(x0 + 1000 * (-b));
			pt1.y = cvRound(y0 + 1000 * (a));
			pt2.x = cvRound(x0 - 1000 * (-b));
			pt2.y = cvRound(y0 - 1000 * (a));
			line(cdst, pt1, pt2, Scalar(0, 0, 255), 1, LINE_AA);
		}
		// Probabilistic Line Transform
		vector<Vec4i> linesP; // will hold the results of the detection
		HoughLinesP(edge, linesP, 1, CV_PI / 180, 30/2, 10/2, 30/2); // runs the actual detection
		alllines.emplace_back(linesP);
		// Draw the lines
		Scalar linecolor;
		for (size_t i = 0; i < linesP.size(); i++)
		{
			Vec4i Lnp = linesP[i];
			if (Lnp[0] == Lnp[2]) continue;
//			cout << (l[1] - l[3]) << " " << (l[0] - l[2]);
			Point2d pt0(0, h*0.6), pt1(Lnp[0], Lnp[1]), pt2(Lnp[2], Lnp[3]), pt3(0, h - 1);
			double angle = atan((Lnp[1] - Lnp[3])*1.0 / (Lnp[0] - Lnp[2]));
			pt0.x = (pt0.y-pt1.y)*1.0 / tan(angle) + pt1.x;
			pt3.x = (pt3.y-pt2.y)*1.0 / tan(angle) + pt2.x;
			if (angle > 0)
				linecolor = Scalar(0, 0, 255);
			else
				linecolor = Scalar(255, 0, 0);
			line(cdstP, pt1, pt2, linecolor, 3, LINE_AA);
			line(cdstP, pt0, pt3, Scalar(0, 255, 0), 1, LINE_4);
		}
		//Show results
		if (bshowmid) {
			imshow("0", original_small[i]);
			imshow("1", edge);
			imshow("2", cdstP);
			waitKey(0);
		}
	}

	//step7, mark the left and right lane area seperately
	vector<double> angle_leftlane, angle_rightlane;
	vector<cv::Mat> mask_leftlane, mask_rightlane;
	for (int i = 0; i < canny_segmented_images.size(); i++) {
		Mat cdst = original_small[i].clone();
		Mat edge = canny_segmented_images[i];
		int w = edge.cols, h = edge.rows;
		Mat maskleft(h, w, CV_8U, cv::Scalar::all(0));
		Mat maskright(h, w, CV_8U, cv::Scalar::all(0));
		Mat leftarea, rightarea;
		double leftangle = 0, rightangle = 0;
		int leftcount = 0, rightcount = 0;
		vector<Vec4i> linesP = alllines[i];
		Scalar colorwhite = Scalar::all(255);
		for (size_t i = 0; i < linesP.size(); i++)
		{
			Vec4i Lnp = linesP[i];
			if (Lnp[0] == Lnp[2]) continue;
			Point2d pt0(0, h*0.6), pt1(Lnp[0], Lnp[1]), pt2(Lnp[2], Lnp[3]), pt3(0, h-1);
			double angle = atan((Lnp[3] - Lnp[1])*1.0 / (Lnp[2] - Lnp[0]));
			pt0.x = (pt0.y - pt1.y)*1.0 / tan(angle) + pt1.x;
			pt3.x = (pt3.y - pt2.y)*1.0 / tan(angle) + pt2.x;
			if (angle < 0) {
				leftangle += angle;
				leftcount++;
				line(maskleft, pt0, pt3, colorwhite, 30, LINE_AA);
				bitwise_and(edge, maskleft, leftarea);
			//	line(cdst, pt0, pt3, Scalar(255, 0, 0), 2, LINE_AA);
			}
			else {
				rightangle += angle;
				rightcount++;
				line(maskright, pt0, pt3, colorwhite, 30, LINE_AA);
				bitwise_and(edge, maskright, rightarea);
			//	line(cdst, pt0, pt3, Scalar(0, 0, 255), 2, LINE_AA);
			}
		}
		leftangle /= leftcount;
		rightangle /= rightcount;
		angle_leftlane.emplace_back(leftangle);
		angle_rightlane.emplace_back(rightangle);
		mask_leftlane.emplace_back(leftarea);
		mask_rightlane.emplace_back(rightarea);
		if (bshowmid) {
			imshow("0", cdst);
			imshow("1", leftarea);
			imshow("2", rightarea);
			waitKey(0);
		}
	}

	//step8, mark the left and right lane area seperately
	vector<cv::Vec4f> leftfittest, rightfittest;
	vector<cv::Vec4f> leftpts, rightpts;
	for (int i = 0; i < original_small.size(); i++) {
		Mat cdst = original_small[i].clone();
		Mat leftarea = mask_leftlane[i];
		Mat rightarea = mask_rightlane[i];
		int w = cdst.cols, h = cdst.rows;
		vector<cv::Point> pointsleft, pointsright;
		double leftangle = angle_leftlane[i], rightangle = angle_rightlane[i];
		cv::Vec4f lineleft, lineright;
		for (int y =h/2; y < h; y++){
			for (int x = 0; x < w; x++) {
				if (leftarea.at<uchar>(y, x) == 255)
					pointsleft.emplace_back(Point(x, y));
				if (rightarea.at<uchar>(y, x) == 255)
					pointsright.emplace_back(Point(x, y));
			}
		}
		cv::fitLine(pointsleft, lineleft, cv::DIST_L1, 1, 0.001, 0.001);
		cv::fitLine(pointsright, lineright, cv::DIST_L1, 1, 0.001, 0.001);
		leftfittest.emplace_back(lineleft);
		rightfittest.emplace_back(lineright);
/*		Point2d pt0(lineleft[2] + lineleft[0] * 150, lineleft[3] + lineleft[1] * 150), pt1(lineleft[2], lineleft[3]);
		Point2d pt2(lineright[2] + lineright[0] * 150, lineright[3] + lineright[1] * 150), pt3(lineright[2], lineright[3]);
		line(cdst, pt0, pt1, Scalar(255, 0, 0), 2, LINE_AA);
		line(cdst, pt2, pt3, Scalar(0, 0, 255), 2, LINE_AA); 
		circle(cdst, pt1, 10, Scalar(0, 255, 0), 2);
		circle(cdst, pt3, 10, Scalar(0, 255, 0), 2);   */
		cv::Vec4f leftpt = drawfittestline(cdst, lineleft, h*0.6, h - 1, false, cv::Scalar(0, 255, 0));
		leftpts.emplace_back(cv::Vec4f(leftpt[0] * 2, leftpt[1] * 2, leftpt[2]*2, leftpt[3]*2));             //recover line pos from small image to big image
		cv::Vec4f rightpt = drawfittestline(cdst, lineright, h*0.6, h - 1, false, cv::Scalar(255, 0, 0));
		rightpts.emplace_back(cv::Vec4f(rightpt[0] * 2, rightpt[1] * 2, rightpt[2] * 2, rightpt[3] * 2));
		if (bshowmid) {
			imshow("0", cdst);
			imshow("1", leftarea);
			imshow("2", rightarea);
			waitKey(0);
		}
	}

	//step9, mark the lane area by left/right line
	for (int i = 0; i < original_images.size(); i++) {
		Mat cdst = original_images[i].clone();
		cv::Vec4f leftpt = leftpts[i];
		cv::Vec4f rightpt = rightpts[i];
		int w = cdst.cols, h = cdst.rows;
		Mat mask(h, w, CV_8U, cv::Scalar::all(0));
		vector< Point> contour;
		contour.push_back(Point(leftpt[0], leftpt[1]));
		contour.push_back(Point(leftpt[2], leftpt[3]));
		contour.push_back(Point(rightpt[2], rightpt[3]));
		contour.push_back(Point(rightpt[0], rightpt[1]));
		const Point *pts = (const cv::Point*) Mat(contour).data;
		int npts = Mat(contour).rows;
		fillPoly(mask, &pts, &npts, 1, cv::Scalar::all(255));	// draw the polygon 
		vector<Mat> spl;
		cv::split(cdst, spl);
		spl[0].setTo(Scalar::all(55), mask);  //green
		spl[2].setTo(Scalar::all(55), mask);  //green
		cv::merge(spl, cdst);
		imshow("0", cdst);
		imshow("1", mask);
		imshow("2", spl[1]);
		waitKey(50);
	}
	return 0;
}

Vec4f drawfittestline(Mat img, Vec4f line_para, int p1, int p2, bool bfx, Scalar color) {
	// Get point oblique point and slope
	Point point0(line_para[2], line_para[3]);
	double k = line_para[1] / line_para[0]; //slope
	Point point1, point2;
	if (bfx)  //from x to y
	{
		//calculate the endpoint of the line :: y = k(x - x0) + y0
		point1.x = p1;
		point1.y = k * (p1 - point0.x) + point0.y;
		point2.x = p2;
		point2.y = k * (p2 - point0.x) + point0.y;
	}
	else {  // from y to x
		//calculate the endpoint of the line :: x = (y - y0)/k + x0 
		point1.y = p1;
		point1.x = (p1 - point0.y) / k + point0.x;
		point2.y = p2;
		point2.x = (p2 - point0.y) / k + point0.x;
	}
	line(img, point1, point2, color, 2, 8, 0);
	Vec4f pts(point1.x, point1.y, point2.x, point2.y);
	return pts;
}
