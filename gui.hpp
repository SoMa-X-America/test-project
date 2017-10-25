#ifndef OPENPOSE_GUI_GUI_HPP
#define OPENPOSE_GUI_GUI_HPP

#include <atomic>
#include <opencv2/core/core.hpp> // cv::Mat
#include <openpose/core/common.hpp>
#include <openpose/gui/enumClasses.hpp>
#include <openpose/gui/frameDisplayer.hpp>
#include <openpose/pose/poseExtractor.hpp>
#include <openpose/pose/poseRenderer.hpp>


namespace op
{

	class data_set {
	public:
		int x[18] = { 0, }, y[18] = { 0, };
		int degree0_1_2 = 0, degree1_2_3 = 0, degree2_3_4 = 0; // 상반신 왼쪽
		int degree0_1_5 = 0, degree1_5_6 = 0, degree5_6_7 = 0; // 상반신 오른쪽
		int degree1_8_9 = 0, degree8_9_10 = 0; // 하반신 왼쪽
		int degree1_11_12 = 0, degree11_12_13 = 0; // 하반신 오른쪽

		int pvecLC[2], pvecCR[2];
		char body_point[18][12] = { "Head", "Neck",
			"RShoulder","RElbow","RWrist",
			"LShoulder","LElbow","LWrist",
			"RHip","RKnee","RAnkle",
			"LHip","LKnee","LAnkle",
			"Chest","Background" };



		data_set(cv::Mat cvMatPoses) {
			if (cvMatPoses.empty()) {
				// 상반신 왼쪽
				degree0_1_2 = -999;
				degree1_2_3 = -999;
				degree2_3_4 = -999;


				// 상반신 오른쪽
				degree0_1_5 = -999;
				degree1_5_6 = -999;
				degree5_6_7 = -999;


				// 하반신 왼쪽
				degree1_8_9 = -999;
				degree8_9_10 = -999;


				// 하반신 오른쪽
				degree1_11_12 = -999;
				degree11_12_13 = -999;

			}
			else {
				for (int i = 0; i < 54; i += 3) {
					x[i / 3] = (int)(cvMatPoses.at<float>(i));
					y[i / 3] = (int)(cvMatPoses.at<float>(i + 1));
				}
				// 상반신 왼쪽
				degree0_1_2 = cal_angle(x[0], y[0], x[1], y[1], x[2], y[2]);
				degree1_2_3 = cal_angle(x[1], y[1], x[2], y[2], x[3], y[3]);
				degree2_3_4 = cal_angle(x[2], y[2], x[3], y[3], x[4], y[4]);


				// 상반신 오른쪽
				degree0_1_5 = cal_angle(x[0], y[0], x[1], y[1], x[2], y[2]);
				degree1_5_6 = cal_angle(x[1], y[1], x[5], y[5], x[6], y[6]);
				degree5_6_7 = cal_angle(x[5], y[5], x[6], y[6], x[7], y[7]);


				// 하반신 왼쪽
				degree1_8_9 = cal_angle(x[1], y[1], x[8], y[8], x[9], y[9]);
				degree8_9_10 = cal_angle(x[8], y[8], x[9], y[9], x[10], y[10]);


				// 하반신 오른쪽
				degree1_11_12 = cal_angle(x[1], y[1], x[11], y[11], x[12], y[12]);
				degree11_12_13 = cal_angle(x[11], y[11], x[12], y[12], x[13], y[13]);
			}
			
		}

		int cal_angle(int Lx, int Ly, int Cx, int Cy, int Rx, int Ry) {
			float vecCL[4] = { Lx - Cx, Cy - Ly, 0 };
			float vecCR[4] = { Rx - Cx, Cy - Ry, 0 };



			pvecLC[0] = vecCL[0];
			pvecLC[1] = vecCL[1];
			pvecCR[0] = vecCR[0];
			pvecCR[1] = vecCR[1];

			//if (vec32[0] != 0 && vec34[0] != 0 && vec32[1] != 0 && vec34[1] != 0) {
			vecCL[2] = sqrt(pow(vecCL[0], 2) + pow(vecCL[1], 2));
			vecCR[2] = sqrt(pow(vecCR[0], 2) + pow(vecCR[1], 2));

			if (vecCL[2] == 0 || vecCR[2] == 0 || vecCL[2]* vecCR[2]==0) return -999;

			float dot_product = (vecCL[0] * vecCR[0] + vecCL[1] * vecCR[1]) / (vecCL[2] * vecCR[2]);
			float cross_product = (vecCL[0] * vecCR[1]) - (vecCL[1] * vecCR[0]);
			float tmp_angle;

			tmp_angle = acos(dot_product) * 180.0 / 3.14159265;
			if (cross_product < 0) tmp_angle *= -1;

			return tmp_angle;
		}

	};



	class OP_API Gui
	{
	public:
		Gui(const bool fullScreen, const Point<int>& outputSize, const std::shared_ptr<std::atomic<bool>>& isRunningSharedPtr,
			const std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>>& videoSeekSharedPtr = nullptr,
			const std::vector<std::shared_ptr<PoseExtractor>>& poseExtractors = {}, const std::vector<std::shared_ptr<PoseRenderer>>& poseRenderers = {});

		void initializationOnThread();

		void update(const cv::Mat& cvOutputData = cv::Mat());

	private:
		// Frames display
		FrameDisplayer mFrameDisplayer;
		// Other variables
		std::vector<std::shared_ptr<PoseExtractor>> mPoseExtractors;
		std::vector<std::shared_ptr<PoseRenderer>> mPoseRenderers;
		std::shared_ptr<std::atomic<bool>> spIsRunning;
		std::shared_ptr<std::pair<std::atomic<bool>, std::atomic<int>>> spVideoSeek;
	};
}

#endif // OPENPOSE_GUI_GUI_HPP
