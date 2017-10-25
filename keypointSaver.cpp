#include <openpose/filestream/fileStream.hpp>
#include <openpose/filestream/keypointSaver.hpp>
#include <iostream>
#include <../include/openpose/gui/gui.hpp>
extern int count = 0;
namespace op
{
    KeypointSaver::KeypointSaver(const std::string& directoryPath, const DataFormat format) :
        FileSaver{directoryPath},
        mFormat{format}
    {
    }

    void KeypointSaver::saveKeypoints(const std::vector<Array<float>>& keypointVector, const std::string& fileName, const std::string& keypointName) const
    {
        try
        {
			if (!keypointVector.empty())
			{
				// File path (no extension)

				const auto fileNameNoExtension = getNextFileName("test") + "_" + keypointName;
				//ȣ��
				const auto realname = getNextFileName(fileName);
				// Get vector of people poses
				std::vector<cv::Mat> cvMatPoses(keypointVector.size());

				for (auto i = 0; i < keypointVector.size(); i++) {
					cvMatPoses[i] = keypointVector[i].getConstCvMat();
				}

				std::vector<cv::Mat> cvMatJointPoses(keypointVector.size());
				for (auto i = 0; i < keypointVector.size(); i++) {
					cvMatJointPoses[i] = keypointVector[i].getConstCvMat();
				}

				for (auto i = 0; i < keypointVector.size(); i++) {
					cvMatJointPoses[i].resize(1);
					//cvMatJointPoses[i].create(10, 1, CV_64F);
				}

				int peopleNum = cvMatJointPoses[0].size.p[1] * cvMatJointPoses[0].dims*cvMatJointPoses[0].size.p[0] / 54;
				int parameterNum = cvMatJointPoses[0].size.p[1];
				int elementNum = cvMatJointPoses[0].dims;
				int angleNum = 10;
				printf("%d ��°������\n", ::count);
				op::data_set data(cvMatJointPoses[0]);
				for (auto i = 0; i < peopleNum*parameterNum*elementNum; i++) {
					switch (i) {
					case 0:
						cvMatJointPoses[0].at<float>(i) = data.degree0_1_2;
						printf("�� ��Ŀ� ���� �� %d��° %f\n", i, cvMatJointPoses[0].at<float>(i));
						break;
					case 1:
						cvMatJointPoses[0].at<float>(i) = data.degree1_2_3;
						printf("�� ��Ŀ� ���� �� %d��° %f\n", i, cvMatJointPoses[0].at<float>(i));
						break;
					case 2:
						cvMatJointPoses[0].at<float>(i) = data.degree2_3_4;
						printf("�� ��Ŀ� ���� �� %d��° %f\n", i, cvMatJointPoses[0].at<float>(i));
						break;
					case 3:
						cvMatJointPoses[0].at<float>(i) = data.degree0_1_5;
						printf("�� ��Ŀ� ���� �� %d��° %f\n", i, cvMatJointPoses[0].at<float>(i));
						break;
					case 4:
						cvMatJointPoses[0].at<float>(i) = data.degree1_5_6;
						printf("�� ��Ŀ� ���� �� %d��° %f\n", i, cvMatJointPoses[0].at<float>(i));
						break;
					case 5:
						cvMatJointPoses[0].at<float>(i) = data.degree5_6_7;
						printf("�� ��Ŀ� ���� �� %d��° %f\n", i, cvMatJointPoses[0].at<float>(i));
						break;
					case 6:
						cvMatJointPoses[0].at<float>(i) = data.degree1_8_9;
						printf("�� ��Ŀ� ���� �� %d��° %f\n", i, cvMatJointPoses[0].at<float>(i));
						break;
					case 7:
						cvMatJointPoses[0].at<float>(i) = data.degree8_9_10;
						printf("�� ��Ŀ� ���� �� %d��° %f\n", i, cvMatJointPoses[0].at<float>(i));
						break;
					case 8:
						cvMatJointPoses[0].at<float>(i) = data.degree1_11_12;
						printf("�� ��Ŀ� ���� �� %d��° %f\n", i, cvMatJointPoses[0].at<float>(i));
						break;
					case 9:
						cvMatJointPoses[0].at<float>(i) = data.degree11_12_13;
						printf("�� ��Ŀ� ���� �� %d��° %f\n", i, cvMatJointPoses[0].at<float>(i));
						break;
					default:
						cvMatJointPoses[0].at<float>(i) = 0;
						break;
					}

				}

				printf("%d��° ����\n", ::count);

			

				// Get names inside file
                std::vector<std::string> keypointVectorNames(cvMatPoses.size());
				for (auto i = 0; i < cvMatPoses.size(); i++) {
					keypointVectorNames[i] = { keypointName + "_" + std::to_string(::count) };
					::count++;
				}
				//ȣ��
				std::vector<std::string> filenames(cvMatPoses.size());
				for(auto i = 0; i < cvMatPoses.size(); i++) {
					filenames[i] = { realname };
				}
                // Record people poses in desired format
				saveData(cvMatJointPoses, keypointVectorNames, fileNameNoExtension, mFormat,realname,filenames,count);
				
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
