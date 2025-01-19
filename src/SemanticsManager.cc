#include "SemanticsManager.h"
#include <thread>

namespace ORB_SLAM2
{

SemanticsManager::SemanticsManager(Map* pMap, const string &strDataset, const string &strSettingPath)
    : mpMap(pMap), mStrDataset(strDataset)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    // add image dimenstions
    mnImgWidth = fSettings["Camera.W"];
    mnImgHeight = fSettings["Camera.H"];

    //RO-MAP
    //Line detect--------------------------------------------------------
    cout<<endl<<"Load RO-MAP Parameters..."<<endl;
    int ExtendBox = fSettings["ExtendBox"];
    mbExtendBox = ExtendBox;
    cout<<"ExtendBox: "<<mbExtendBox<<endl;

    int CheckBoxEdge = fSettings["CheckBoxEdge"];
    mbCheckBoxEdge = CheckBoxEdge;
    cout<<"CheckBoxEdge: "<<CheckBoxEdge<<endl;

    cv::FileNode node = fSettings["IgnoreCategory"];
    cout<<"IgnoreCategory: ";
    for(auto it = node.begin();it!=node.end();it++)
    {
        int number = *it;
        mvIgnoreCategory.insert(number);
        cout<<number<<" ";
    }
    cout<<endl;
    mnBoxMapPoints = fSettings["BoxMapPoints"];
    cout<<"BoxMapPoints: "<<mnBoxMapPoints<<endl;
    if(mnBoxMapPoints < 1)
    {
        cerr<<"Failed to load RO-MAP parameters, Please add parameters to yaml file..."<<endl;
        exit(0);
    }

    mnMinimumContinueObs = fSettings["Minimum.continue.obs"];
    cout<<"MinimumContinueObs: "<<mnMinimumContinueObs<<endl;

    AddMPsDistMultiple = fSettings["Add.MPs.distance.multiple"];
    cout<<"AddMPsDistMultiple: "<<AddMPsDistMultiple<<endl;

    int numoctaves = 1;
    float octaveratio = 2.0;
    bool use_LSD = false;  // use LSD detector or edline detector    
    float line_length_thres = 15;
    mpLineDetect = new line_lbd_detect(numoctaves,octaveratio);

    mpLineDetect->use_LSD = use_LSD;
    mpLineDetect->line_length_thres = line_length_thres;
    mpLineDetect->save_imgs = false;
    mpLineDetect->save_txts = false;
    
    //read t-test data
    ifstream f;
    f.open("./lib/t_test.txt");
    if(!f.is_open())
    {
        cerr<<"Can't read t-test data"<<endl;
        exit(0);
    }
    for(int i=0;i<101;i++)
    {
        for(int j=0;j<4;j++)
            f >> tTest[i][j];  
    }
    f.close();
}

void SemanticsManager::AddTaskToQueue(Task task)
{
    std::unique_lock<std::mutex> lock(mMutex);
    mTaskQueue.push(task);
}

void SemanticsManager::Run()
{
    while(1)
    {
        if(mTaskQueue.empty())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        std::unique_lock<std::mutex> lock(mMutex);
        Task currentTask = mTaskQueue.front();
        Frame* pFrame = currentTask.frame;
        const cv::Mat mImGray = currentTask.imgGray;
        const cv::Mat mImgInstance = currentTask.imgInstance;
        mTaskQueue.pop();
        lock.unlock();

        // Frame processing code
        vector<Bbox> Bboxs;
        {
            //offline read object detection----------------------------
            string sYolopath = mstrDataset + "/" + "bbox/" + to_string(mCurrentFrame.mTimeStamp) + ".txt";
            
            ifstream f;
            f.open(sYolopath);
            if(!f.is_open())
            {
                cout << "yolo_detection file open fail" << endl;
                exit(0);
            }
            
            //Read txt
            string line;
            float num;
            vector<float> row;
            while(getline(f,line))
            {
                stringstream s(line);
                while (s >> num)
                {
                    row.push_back(num);
                }
                Bbox newBbox;
                newBbox.mnClass = row[0];

                //extend box
                if(mbExtendBox)
                {
                    newBbox.x = max(0.0f,row[1] - 10);
                    newBbox.y = max(0.0f,row[2] - 10);
                    newBbox.width = min(float(mImgWidth-1) - newBbox.x,row[3] + 20);
                    newBbox.height = min(float(mImgHeight-1) - newBbox.y,row[4] + 20); 
                }
                else
                {
                    newBbox.x = row[1];
                    newBbox.y = row[2];
                    newBbox.width = row[3];
                    newBbox.height = row[4];
                }
                newBbox.mfConfidence = row[5];
                Bboxs.push_back(newBbox);
                row.clear();
            }
            f.close();

        }
        
        /*Filter bad Bbox. Including the following situations:
            Close to image edge
            Overlap each other
            Bbox is too large
            */
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> objectClouds;
        if(!Bboxs.empty())
        {
            vector<int> resIdx(Bboxs.size(),1);
            vector<Bbox> resBbox;
            for(size_t i=0;i<Bboxs.size();i++)
            {
                if(!resIdx[i])
                    continue;

                Bbox &box = Bboxs[i];

                // There will be errors in target detection, and error categories can be filtered here
                if(mvIgnoreCategory.find(box.mnClass) != mvIgnoreCategory.end())
                {
                    resIdx[i] = 0;
                    continue;
                }

                if(mbCheckBoxEdge)
                {
                    //Close to image edge
                    if(box.x < 20 || box.x+box.width > mImgWidth-20 || box.y < 20 || box.y+box.height > mImgHeight-20)
                    {  
                        if(box.area() < mImgWidth * mImgHeight * 0.05)
                        {
                            resIdx[i] = 0;
                            continue;
                        }
                        box.mbEdge = true;
                        if(box.area() < mImgWidth * mImgHeight * 0.1)
                            box.mbEdgeAndSmall = true;
                    }
                }

                //Bbox is large than half of img
                if(box.area() > mImgWidth * mImgHeight * 0.5)
                {
                    resIdx[i] = 0;
                    continue;
                }
                else if(box.area() < mImgWidth * mImgHeight * 0.005)
                {
                    resIdx[i] = 0;
                    continue;
                }

                //Overlap each other
                for(size_t j=0;j<Bboxs.size();j++)
                {
                    if(i == j || resIdx[j] == 0)
                        continue;
                    
                    Bbox &box_j = Bboxs[j];
                    float SizeScale = min(box_j.area(),box.area()) / max(box_j.area(),box.area());
                    if(SizeScale > 0.25)
                    {
                        float overlap = (box & box_j).area();
                        float IOU = overlap / (box.area() + box_j.area() - overlap);
                        if(IOU > 0.4)
                        {
                            resIdx[i] = 0;
                            resIdx[j] = 0;
                            break;
                        }
                    }
                } 
            }

            for(size_t i=0;i<Bboxs.size();i++)
            {
                if(resIdx[i])
                {
                    resBbox.push_back(Bboxs[i]);

                    // // get the pointcloud for the object
                    // pcl::PointCloud<pcl::PointXYZ>::Ptr objectCloud(new pcl::PointCloud<pcl::PointXYZ>);
                    // const string sCloudpath = strDataset + "/pointclouds/" + to_string(timestamp) + "/" + to_string(i) + ".ply";
                    // pcl::io::loadPLYFile(sCloudpath, *objectCloud);
                    // objectClouds.push_back(objectCloud);

                }
            }
            Bboxs = resBbox;
        }
        
        if(!Bboxs.empty())
        {
            mCurrentFrame.mbDetectObject = true;
            mCurrentFrame.mvBbox = Bboxs;
            mCurrentFrame.UndistortFrameBbox();

            //Line detect-----------------------------------------------
            //using distort Img
            cv::Mat undistortImg = mImGray.clone();
            if(mDistCoef.at<float>(0)!=0.0)
            {
                cv::undistort(mImGray,undistortImg,mK,mDistCoef);
            }
            mpLineDetect->detect_raw_lines(undistortImg,mCurrentFrame.mvLines);
            
            vector<KeyLine> FilterLines;
            mpLineDetect->filter_lines(mCurrentFrame.mvLines,FilterLines);
            mCurrentFrame.mvLines = FilterLines;
            Eigen::MatrixXd LinesEigen(FilterLines.size(),4);
            for(size_t i=0;i<FilterLines.size();i++)
            {
                LinesEigen(i,0)=FilterLines[i].startPointX;
                LinesEigen(i,1)=FilterLines[i].startPointY;
                LinesEigen(i,2)=FilterLines[i].endPointX;
                LinesEigen(i,3)=FilterLines[i].endPointY;
            }
            mCurrentFrame.mLinesEigen = LinesEigen;
            
            //creat object_Frame---------------------------------------------------
            for(size_t i=0;i<mCurrentFrame.mvBboxUn.size();i++)
            {
                Object_Frame object_frame;
                object_frame.mnFrameId = mCurrentFrame.mnId;
                object_frame.mBbox = mCurrentFrame.mvBboxUn[i];
                object_frame.mnClass = object_frame.mBbox.mnClass;
                object_frame.mfConfidence = object_frame.mBbox.mfConfidence;
                mCurrentFrame.mvObjectFrame.push_back(object_frame);
            }
        }
        
        //Assign feature points and lines to detected objects
        mCurrentFrame.AssignFeaturesToBbox(mImgInstance);
        mCurrentFrame.AssignLinesToBbox();
    }
}
} // namespace ORB_SLAM2