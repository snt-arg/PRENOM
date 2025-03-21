/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* Modification: PRENOM
* Version: 1.0
* Created: 12/25/2024
* Author: Saad Ejaz
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <unistd.h>
#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames,
                vector<string> &vstrInstanceFilenames, vector<string> &vstrDepthFilenames,
                vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_nerf_config path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<string> vstrInstanceFilenames;
    vector<string> vstrDepthFilenames;
    vector<double> vTimestamps;
    const string strDataset = string(argv[3]);
    const string strFile = string(argv[3]) + "/img.txt";
    const string strDatasetConfig = string(argv[3]) + "/config.yaml";
    const string strNeRFConfig = string(argv[2]);

    LoadImages(strFile, vstrImageFilenames,vstrInstanceFilenames, vstrDepthFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],strDatasetConfig,strNeRFConfig,strDataset,nImages,ORB_SLAM2::System::RGBD,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat im;
    cv::Mat imgInstance;
    cv::Mat imgDepth;
 
    // sleep for some time to wait for the system to initialize
    usleep(1*1e6);

    auto start_time = std::chrono::high_resolution_clock::now();
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image from file
        im = cv::imread(strDataset+"/"+vstrImageFilenames[ni],cv::IMREAD_UNCHANGED);
        imgInstance = cv::imread(strDataset+"/"+vstrInstanceFilenames[ni],cv::IMREAD_UNCHANGED);
        imgDepth = cv::imread(strDataset+"/"+vstrDepthFilenames[ni],cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << strDataset << "/" << vstrImageFilenames[ni] << endl;
            return 1;
        }
        if(imgInstance.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << strDataset << "/" << vstrInstanceFilenames[ni] << endl;
            return 1;
        }
        if(imgDepth.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << strDataset << "/" << vstrDepthFilenames[ni] << endl;
            return 1;
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        // Pass the image to the SLAM system
        SLAM.TrackRGBD(im,imgDepth,imgInstance,tframe,strDataset);
        
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        // std::cout << "Tracking time: " << ttrack * 1000<< "ms" << std::endl;

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        double limitT = 0;
        if(ni<nImages-1)
        {
            T = vTimestamps[ni+1]-tframe;
            limitT = vTimestamps[ni+1];
        }
        else if(ni>0)
        {
            T = tframe-vTimestamps[ni-1];
            limitT = tframe;
        }

        // minimum time to be slept is until the next timestamp
        auto current_time = std::chrono::high_resolution_clock::now();
        double current_ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(current_time - start_time).count();
        double min_time_to_sleep = limitT - current_ttrack;
        if (min_time_to_sleep > 0)
            usleep(min_time_to_sleep*1e6);
    }

    auto pre_end_time = std::chrono::high_resolution_clock::now();
    double pre_ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(pre_end_time - start_time).count();
    std::cout << "Pre-Finish time: " << pre_ttrack<< "s" << std::endl;

    usleep(1*1e6);
    SLAM.FinishNeRFs();
    auto end_time = std::chrono::high_resolution_clock::now();
    double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(end_time - start_time).count();
    std::cout << "Total time: " << ttrack<< "s" << std::endl;
    
    cout<<endl<<"Press Enter to save NeRFs (render images and obj.ply) or Crtl+C to quit ..." <<endl;
    getchar();

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    SLAM.SaveTrajectoryTUM("FrameTrajectory.txt");
    SLAM.SaveObjects("./output/objects.txt");
    
    return 0;
}

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames,vector<string> &vstrInstanceFilenames, vector<string> &vstrDepthFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    // skip first three lines
    string s0;
    getline(f,s0);

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sName;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sName;
            string rgb = "rgb/" + sName;
            string instance = "instance/" +  sName;
            string depth = "depth/" + sName;
            vstrImageFilenames.push_back(rgb);
            vstrInstanceFilenames.push_back(instance);
            vstrDepthFilenames.push_back(depth);
        }
    }
    f.close();       
}
