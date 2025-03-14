/**
* Modification: PRENOM
* Version: 1.0
* Created: 01/27/2025
* Author: Saad Ejaz
*/

#include "ObjectManager.h"
#include <thread>

#define GRID_SIZE_CUBE 262144

namespace ORB_SLAM2
{

ObjectManager::ObjectManager(Map* pMap, const string &strDataset, const string &strSettingPath)
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

    // Precompute inverse of the intrinsic matrix
    Eigen::Matrix3f eigenK;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            eigenK(i, j) = mK.at<float>(i, j);
    mEigenInvK = eigenK.inverse();

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

    // Line detect--------------------------------------------------------
    cout<<endl<<"Load Parameters..."<<endl;
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
        cerr<<"Failed to load parameters, Please add parameters to yaml file..."<<endl;
        exit(0);
    }

    mfMaxDepth = fSettings["MaxDepth"];
    cout<<"MaxDepth: "<<mfMaxDepth<<endl;

    mnMinimumContinueObs = fSettings["Minimum.continue.obs"];
    cout<<"MinimumContinueObs: "<<mnMinimumContinueObs<<endl;

    mnMaxBoxPercent = fSettings["MaxBoxPercent"];
    cout<<"MaxBoxPercent: "<<mnMaxBoxPercent<<endl;

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

    // load all priors for the objects
    std::ifstream file("cookbook/recipes.txt");
    if (!file) {
        std::cerr << "Error opening recipes file" << std::endl;
        return;
    }
    // fill the mvAvailableClasses
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int classId;
        iss >> classId;
        mvAvailableClasses.push_back(classId);
        continue;
    }
    file.close();
    cout << "Available classes: ";
    for (int& classId : mvAvailableClasses) {
        cout << classId << " ";
    }
    cout << endl;

    // read the available object configs
    for (int& classId : mvAvailableClasses) {
        std::string filename = "cookbook/" + std::to_string(classId) + "/config.json";
        std::ifstream file(filename);
        if (!file) {
            std::cerr << "Error opening config file for class " << classId << std::endl;
            continue;
        }
        nlohmann::json j;
        file >> j;

        // create the object config
        ObjectConfig config;
        config.classId = classId;
        config.isKnown = classId != 0;
        config.align.isDensityBased = j["align"]["is_density_based"];
        config.align.polyFit = j["align"]["poly_fit"];
        config.align.metricResolution = j["align"]["metric_resolution"];
        config.align.normalizedResolution = j["align"]["normalized_resolution"];
        config.align.numSampleAngles = j["align"]["num_sample_angles"];
        config.icp.enabled = j["icp"]["enabled"];
        config.icp.maxIterations = j["icp"]["max_iterations"];
        config.icp.maxTransPercent = j["icp"]["max_trans_percent"];
        config.icp.minRotCos = j["icp"]["min_rot_cos"];
        config.symmetry.isReflectional = j["symmetry"]["is_reflectional"];
        config.symmetry.isRotational = j["symmetry"]["is_rotational"];
        config.bbox.expand = j["bbox"]["expand"];
        config.bbox.incrementX = j["bbox"]["increment_x"];
        config.bbox.incrementY = j["bbox"]["increment_y"];
        config.bbox.incrementZ = j["bbox"]["increment_z"];
        config.pointcloud.maxPoints = j["pointcloud"]["max_points"];
        config.pointcloud.minPoints = j["pointcloud"]["min_points"];
        config.downsample.voxelSize = j["downsample"]["voxel_size"];
        config.downsample.minPointsPerVoxel = j["downsample"]["min_points_per_voxel"];
        config.clustering.enabled = j["clustering"]["enabled"];
        config.clustering.tolerance = j["clustering"]["tolerance"];
        config.outlierRemoval.enabled = j["outlier_removal"]["enabled"];
        config.outlierRemoval.minNeighbors = j["outlier_removal"]["min_neighbors"];
        config.outlierRemoval.stdDev = j["outlier_removal"]["std_dev"];
        config.pointcloudEIF.enabled = j["pointcloud_eif"]["enabled"];
        config.pointcloudEIF.threshold = j["pointcloud_eif"]["threshold"];
        config.centroidTTest.enabled = j["centroid_t_test"]["enabled"];
        config.centroidTTest.minHistorySize = j["centroid_t_test"]["min_history_size"];
        config.rankSumTest.enabled = j["rank_sum_test"]["enabled"];
        config.rankSumTest.minHistorySize = j["rank_sum_test"]["min_history_size"];
        mmObjectConfigs[classId] = config;

        // defaults only have configs
        if (classId == 0)
            continue;

        // load the prior type based on the config
        if (config.align.isDensityBased)
        {
            // to get the density prescale, read the network.json file
            std::ifstream networkFile("cookbook/" + std::to_string(classId) + "/network.json");
            if (!networkFile) {
                std::cerr << "Error opening network file" << std::endl;
                continue;
            }
            nlohmann::json network;
            networkFile >> network;
            const float prescale = network["misc"]["depth_prescale"];

            // load the density grid
            std::string densityPath = "cookbook/" + std::to_string(classId) + "/density.ply";
            mmObjectDensities[classId] = new float[GRID_SIZE_CUBE];
            // float* densityGrid = densities;
            std::ifstream densityFile(densityPath);
            if (!densityFile) {
                std::cerr << "Error opening density file" << std::endl;
                continue;
            }

            std::string line;
            while (std::getline(densityFile, line)) {
                if (line == "end_header")
                    break;
            }

            // read the density values
            size_t index = 0;
            while (std::getline(densityFile, line)) {
                std::istringstream iss(line);
                float density;
                if (!(iss >> density >> density >> density >> density)) {
                    break;
                }
                mmObjectDensities[classId][index++] = exp(density/prescale);
            }
        }

        // load the normalized mesh model
        pcl::PointCloud<pcl::PointXYZ>::Ptr model(new pcl::PointCloud<pcl::PointXYZ>);
        std::string modelPath = "cookbook/" + std::to_string(classId) + "/model.ply";
        if (pcl::io::loadPLYFile<pcl::PointXYZ>(modelPath, *model) == -1) {
            std::cerr << "Error loading model file for class " << classId << std::endl;
            continue;
        }
        model->width = model->size();
        model->height = 1;
        model->is_dense = false;
        mmObjectModels[classId] = model;
    }
}

void ObjectManager::AddTaskToQueue(Task task)
{
    std::unique_lock<std::mutex> lock(mMutex);
    mTaskQueue.push(task);
}

void ObjectManager::Run()
{
    while(1)
    {
        if(mTaskQueue.empty())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        auto task_start = std::chrono::high_resolution_clock::now();

        // get the next task
        std::unique_lock<std::mutex> lock(mMutex);
        Task currentTask = mTaskQueue.front();
        Frame* pFrame = currentTask.frame;
        KeyFrame* pKF = currentTask.keyframe;
        const cv::Mat imColor = currentTask.imgColor;
        const cv::Mat imGray = currentTask.imgGray;
        const cv::Mat imgInstance = currentTask.imgInstance;
        const cv::Mat imgDepth = currentTask.imgDepth;
        Eigen::Matrix4f Twc = Converter::toMatrix4f(pFrame->mTcw).inverse();
        if (pKF)
            Twc = Converter::toMatrix4f(pKF->GetPoseInverse());
        mTaskQueue.pop();
        lock.unlock();

        // Process the task
        vector<Bbox> Bboxs;
        {
            //offline read object detection----------------------------
            string sYolopath = mStrDataset + "/" + "bbox/" + to_string(pFrame->mTimeStamp) + ".txt";
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
                    newBbox.width = min(float(mnImgWidth-1) - newBbox.x,row[3] + 20);
                    newBbox.height = min(float(mnImgHeight-1) - newBbox.y,row[4] + 20); 
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
                    if(box.x < 20 || box.x+box.width > mnImgWidth-20 || box.y < 20 || box.y+box.height > mnImgHeight-20)
                    {  
                        if(box.area() < mnImgWidth * mnImgHeight * 0.05)
                        {
                            resIdx[i] = 0;
                            continue;
                        }
                        box.mbEdge = true;
                        if(box.area() < mnImgWidth * mnImgHeight * 0.1)
                            box.mbEdgeAndSmall = true;
                    }
                }

                // Bbox is large than a certain percentage of the image
                if(box.area() > mnImgWidth * mnImgHeight * mnMaxBoxPercent)
                {
                    resIdx[i] = 0;
                    continue;
                }
                else if(box.area() < mnImgWidth * mnImgHeight * 0.005)
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
                    resBbox.push_back(Bboxs[i]);
            }
            Bboxs = resBbox;
        }
        
        if(!Bboxs.empty())
        {
            pFrame->mbDetectObject = true;
            pFrame->mvBbox = Bboxs;
            pFrame->UndistortFrameBbox();

            //Line detect-----------------------------------------------
            //using distort Img
            cv::Mat undistortImg = imGray.clone();
            if(mDistCoef.at<float>(0)!=0.0)
            {
                cv::undistort(imGray,undistortImg,mK,mDistCoef);
            }
            mpLineDetect->detect_raw_lines(undistortImg,pFrame->mvLines);
            
            vector<KeyLine> FilterLines;
            mpLineDetect->filter_lines(pFrame->mvLines,FilterLines);
            pFrame->mvLines = FilterLines;
            Eigen::MatrixXd LinesEigen(FilterLines.size(),4);
            for(size_t i=0;i<FilterLines.size();i++)
            {
                LinesEigen(i,0)=FilterLines[i].startPointX;
                LinesEigen(i,1)=FilterLines[i].startPointY;
                LinesEigen(i,2)=FilterLines[i].endPointX;
                LinesEigen(i,3)=FilterLines[i].endPointY;
            }
            pFrame->mLinesEigen = LinesEigen;
            
            //creat object_Frame---------------------------------------------------
            for(size_t i=0;i<pFrame->mvBboxUn.size();i++)
            {
                Object_Frame object_frame;
                object_frame.mnFrameId = pFrame->mnId;
                object_frame.mBbox = pFrame->mvBboxUn[i];
                object_frame.mnClass = object_frame.mBbox.mnClass;
                object_frame.mfConfidence = object_frame.mBbox.mfConfidence;
                pFrame->mvObjectFrame.push_back(object_frame);
            }
        }
        
        //Assign feature points and lines to detected objects
        pFrame->AssignFeaturesToBbox(imgInstance);
        pFrame->AssignLinesToBbox();

        // set the class pointclouds
        vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> vClassClouds;
        ClassPointcloudsFromDepth(imgDepth,imgInstance,pFrame->mvObjectFrame,vClassClouds,mfMaxDepth);
        for (size_t i=0; i < pFrame->mvObjectFrame.size(); i++)
        {
            Object_Frame& obj = pFrame->mvObjectFrame[i];
            obj.mCloud = vClassClouds[i]; 
        }

        //object-nerf-SLAM-----------------The main functions are as follows--------------------------------------
        mvPointsToDrawer.clear();

        if(pFrame->mbDetectObject)
        {
            vector<Object_Frame>& Objects = pFrame->mvObjectFrame;
            //1. After optimize pose, associate MapPoints with objects using KeyPoints;
            for(Object_Frame& obj : Objects)
            {   
                obj.mSumPointsPos = cv::Mat::zeros(3,1,CV_32F);
                
                for(const int& i : obj.mvIdxKeyPoints)
                {
                    if(pFrame->mvpMapPoints[i])
                        if(pFrame->mvbOutlier[i])
                            continue;
                        else
                        {
                            obj.mvpMapPoints.push_back(pFrame->mvpMapPoints[i]);
                            cv::Mat MPworldPos = pFrame->mvpMapPoints[i]->GetWorldPos();
                            obj.mSumPointsPos += MPworldPos;

                            // to FrameDraw
                            cv::Point point = pFrame->mvKeysUn[i].pt;
                            mvPointsToDrawer.push_back(point);
                        }
                }
                
                if(obj.mvpMapPoints.size() < mnBoxMapPoints)
                {
                    obj.mbBad = true;
                    continue;
                }

                //on edge 
                if(obj.mBbox.mbEdge)
                    if(obj.mvpMapPoints.size() < mnBoxMapPoints * 2)
                    {
                        obj.mbBad = true;
                        continue;
                    }

                //2. filter mappoints using Boxplot
                if(!obj.mvpMapPoints.empty())
                {
                    obj.FilterMPByBoxPlot(pFrame->mTcw);
                }

                //3. Calculate object  information
                //Calculate the mean and standard deviation
                obj.CalculateMeanAndStandard();

                //Construct Bbox by reprojecting MapPoints, for data association
                obj.ConstructBboxByMapPoints(*pFrame);

            }

            //deal ObjectMap
            unique_lock<mutex> lock(mpMap->mMutexObjects);

            mvNewOrChangedObj.clear();
            //4. first init object map
            if(!mbInitObjectMap && mnFramesPassed > 1)
            {
                if(InitObjectMap(pFrame))
                    mbInitObjectMap = true;
            } 
            else if (mbInitObjectMap)
            {

                //5. data association
                //Please refer to our previous work for theoretical.

                //Two strategies  1. Consecutive association    2. Non-consecutive association

                //update mpLastFrame->mvObjectMap(backend merge object)
                for(size_t i=0;i<mpLastFrame->mvObjectMap.size();i++)
                {
                    if(mpLastFrame->mvObjectMap[i])
                    {
                        Object_Map* pObj = mpLastFrame->mvObjectMap[i];

                        Object_Map* pReObj = pObj->GetReplaced();
                            if(pReObj)
                                mpLastFrame->mvObjectMap[i] = pReObj;
                    }
                }
                

                //Separate two types of Object_Map
                set<Object_Map*> sObjsIF;
                set<Object_Map*> sObjsNIF;
            
                for(Object_Map* pObj : mpLastFrame->mvObjectMap)
                {
                    if(pObj && !pObj->IsBad())
                        sObjsIF.insert(pObj);
                }
                for(Object_Map* pObj : mpMap->GetAllObjectMap())
                {   
                    if(pObj->IsBad())
                        continue;
                    if(sObjsIF.find(pObj) != sObjsIF.end())
                        continue;
                    else
                    {   
                        //Successful inter frame matching for at least three consecutive frames
                        if(pObj->mnObs < mnMinimumContinueObs)
                        {
                            pObj->SetBad("No inter-frame matching");
                            continue;
                        }    

                        sObjsNIF.insert(pObj);
                        //for Non interframe Data Association
                        pObj->ConstructBboxByMapPoints(*pFrame);
                    }
                        
                }
                
                int nObjFrame = pFrame->mvObjectFrame.size();
                pFrame->mvObjectMap = vector<Object_Map*>(nObjFrame,static_cast<Object_Map*>(NULL));

                /* cout<<"sObjsIF.size(): "<<sObjsIF.size()<<endl;
                cout<<"sObjsNIF.size(): "<<sObjsNIF.size()<<endl;
                cout<<"Objects.size(): "<<Objects.size()<<endl; */
                //cout<<"------------------------------------------"<<endl;
                
                for(size_t i=0;i<Objects.size();i++)
                {
                    Object_Frame& obj = Objects[i];
                    if(obj.mbBad)
                        continue;

                    bool bIFass = false;
                    bool bNIFass = false;

                    Object_Map* AssOBJ = static_cast<Object_Map*>(NULL);
                    vector<Object_Map*> possibleOBJ;

                    //1. Consecutive association

                    Object_Map* IoUassOBJ = static_cast<Object_Map*>(NULL);
                    float fMaxIoU = 0;

                    Object_Map* MaxMPassOBJ = static_cast<Object_Map*>(NULL);
                    int nMaxMPs = 0;
                    set<MapPoint*> ObjMPs(obj.mvpMapPoints.begin(),obj.mvpMapPoints.end());
                    
                    for(Object_Map* pObjMap : sObjsIF)
                    {
                        //Inter frame IOU data association
                        if(pObjMap->IsBad())
                            continue;
                        if(pObjMap->mnClass != obj.mnClass)
                            continue;
                        if(pObjMap->mnlatestObsFrameId == pFrame->mnId)
                            continue;

                        cv::Rect predictBbox;
                        //CurrentFrame Bbox prediction
                        if(pObjMap->mLastBbox != pObjMap->mLastLastBbox)
                        {
                            float top_left_x = pObjMap->mLastBbox.x * 2 - pObjMap->mLastLastBbox.x;
                            float top_left_y = pObjMap->mLastBbox.y * 2 - pObjMap->mLastLastBbox.y;
                    
                            if(top_left_x < pFrame->mnMinX)
                                top_left_x = pFrame->mnMinX;
                            if(top_left_y < pFrame->mnMinY)
                                top_left_y = pFrame->mnMinY;  

                            float width = pObjMap->mLastBbox.width * 2 - pObjMap->mLastLastBbox.width;
                            float height = pObjMap->mLastBbox.height * 2 - pObjMap->mLastLastBbox.height;

                            if(width > pFrame->mnMaxX - top_left_x)
                                width = pFrame->mnMaxX - top_left_x;
                            
                            if(height > pFrame->mnMaxY - top_left_y)
                                height = pFrame->mnMaxY - top_left_y;
                            
                            predictBbox = cv::Rect(top_left_x,top_left_y,width,height);

                        }
                        else
                        {
                            predictBbox = pObjMap->mLastBbox;
                        }

                        float IoUarea = (predictBbox & obj.mBbox).area(); 
                        IoUarea = IoUarea / float((predictBbox.area() + obj.mBbox.area() - IoUarea));
                        if(IoUarea > 0.5 && IoUarea > fMaxIoU)
                        {
                            fMaxIoU = IoUarea;
                            IoUassOBJ = pObjMap;  
                        }
                        
                        //Inter frame MapPoints data association
                        int nShareMP = 0;
                        //Data association is not performed when there are too few MapPoints
                        if(ObjMPs.size() > 6)
                        {
                            for(MapPoint* pMP : pObjMap->mvpMapPoints)
                            {
                                if(ObjMPs.find(pMP) != ObjMPs.end())
                                ++nShareMP;
                            }
                        
                            if(nShareMP > ObjMPs.size() / 3 && nShareMP> nMaxMPs)
                            {
                                nMaxMPs = nShareMP;
                                MaxMPassOBJ = pObjMap;
                            }
                        }

                    }
                    
                    //have association
                    if(fMaxIoU > 0.7)
                    {
                        if(IoUassOBJ->whetherAssociation(obj,*pFrame))
                        {
                            AssOBJ = IoUassOBJ;
                            bIFass = true;
                        }
                        else
                            bIFass = false;
                    }
                    else if(fMaxIoU > 0 && nMaxMPs > 0)
                    {   
                        //same association
                        if(IoUassOBJ == MaxMPassOBJ)
                        {
                            if(IoUassOBJ->whetherAssociation(obj,*pFrame))
                            {
                                AssOBJ = IoUassOBJ;
                                bIFass = true;
                            }
                            else
                                bIFass = false;
                        }
                        else
                        {
                            // have association but not same
                            bIFass = false;
                            obj.mbBad = true;
                        }
                    }
                    else if(fMaxIoU == 0 && nMaxMPs==0)
                    {
                        bIFass = false;
                    }
                    else
                    {

                        if(fMaxIoU > 0)
                        {
                            if(IoUassOBJ->whetherAssociation(obj,*pFrame))
                            {
                                AssOBJ = IoUassOBJ;
                                bIFass = true;
                            }
                            else
                                bIFass = false;
                        }
                        else
                        {
                            if(MaxMPassOBJ->whetherAssociation(obj,*pFrame))
                            {
                                AssOBJ = MaxMPassOBJ;
                                bIFass = true;
                            }
                            else
                                bIFass = false;
                        }
                    }
                    
                    // Non-consecutive Association
                    for(Object_Map* pObjMap : sObjsNIF)
                    {   
                        //cout<<"pObjMap IsBad: "<<pObjMap->IsBad()<<endl;
                        if(pObjMap->IsBad() || pObjMap->mnClass != obj.mnClass)
                            continue;

                        if(pObjMap->mnlatestObsFrameId == pFrame->mnId)
                            continue;

                        //Data association is not performed when there are too few MapPoints
                        int nShareMP = 0;
                        if(ObjMPs.size() > 6)
                        {
                            for(MapPoint* pMP : pObjMap->mvpMapPoints)
                            {
                                if(ObjMPs.find(pMP) != ObjMPs.end())
                                    ++nShareMP;
                            }
                        
                            if(nShareMP > ObjMPs.size() / 3)
                            {
                                possibleOBJ.push_back(pObjMap);
                                continue;
                            }
                        }

                        int nobs = pObjMap->mnObs;
                        //t-test 
                        float tx,ty,tz;

                        tx = abs(pObjMap->mHistoryPosMean.at<float>(0) - obj.mPosMean.at<float>(0));
                        tx = sqrt(nobs) * tx / pObjMap->mfPosStandardX;
                        ty = abs(pObjMap->mHistoryPosMean.at<float>(1) - obj.mPosMean.at<float>(1));
                        ty = sqrt(nobs) * ty / pObjMap->mfPosStandardY;
                        tz = abs(pObjMap->mHistoryPosMean.at<float>(2) - obj.mPosMean.at<float>(2));
                        tz = sqrt(nobs) * tz / pObjMap->mfPosStandardZ;
                        // Degrees of freedom.
                        int deg = min(100,nobs-1);

                        if(pObjMap->mnObs > 4)
                        {
                            //0.05
                            float th = tTest[deg][2];
                            if(tx<th && ty<th && tz<th)
                            {
                                possibleOBJ.push_back(pObjMap);
                                continue;
                            }

                        }

                        //check IoU, reproject associate
                        float IoUarea = (pObjMap->mMPsProjectRect & obj.mBbox).area();
                        IoUarea = IoUarea / float((pObjMap->mMPsProjectRect.area() + obj.mBbox.area() - IoUarea));
                        if(IoUarea > 0.3)
                        {   
                            //0.001
                            float th = tTest[deg][4];
                            if(tx<th && ty<th && tz<th)
                            {
                                possibleOBJ.push_back(pObjMap);
                                continue;
                            }
                            else if( (tx+ty+tz) / 3 < 2 * th)
                            {
                                possibleOBJ.push_back(pObjMap);
                                continue;
                            }
                        }
                    }

                    //try possible object
                    if(!bIFass && !possibleOBJ.empty())
                    {
                        sort(possibleOBJ.begin(),possibleOBJ.end(),[](const Object_Map* left,const Object_Map* right){return left->mnObs < right->mnObs;});

                        for(int i=possibleOBJ.size()-1;i>=0;i--)
                        {
                            if(possibleOBJ[i]->whetherAssociation(obj,*pFrame))
                            {
                                AssOBJ = possibleOBJ[i];
                                bNIFass = true;
                                break;
                            }
                        }
                        if(bNIFass)
                        {
                            for(int i=possibleOBJ.size()-1;i>=0;i--)
                            {
                                if(possibleOBJ[i] == AssOBJ)
                                {
                                    continue;
                                }
                                //judge in the backend
                                AssOBJ->mPossibleSameObj[possibleOBJ[i]]++;
                            }
                        }
                    }
                    else if(!possibleOBJ.empty())
                    {
                        for(int i=possibleOBJ.size()-1;i>=0;i--)
                            {
                                if(possibleOBJ[i] == AssOBJ)
                                {
                                    continue;
                                }
                                //judge in the backend
                                AssOBJ->mPossibleSameObj[possibleOBJ[i]]++;
                            }
                    }

                    //update 2D information
                    if(bIFass || bNIFass)
                    {
                        //cout<<"bIFass: "<<bIFass<<" bNIFass: "<<bNIFass<<endl;
                        AssOBJ->mnlatestObsFrameId = pFrame->mnId;
                        AssOBJ->mnObs++;
                        if(bIFass)
                            AssOBJ->mLastLastBbox = AssOBJ->mLastBbox;
                        else
                            AssOBJ->mLastLastBbox = obj.mBbox;
                        AssOBJ->mLastBbox = obj.mBbox;
                        AssOBJ->mlatestFrameLines = obj.mLines;
                        AssOBJ->mvHistoryPos.push_back(obj.mPosMean);
                        
                        bool checkMPs = false;
                        SE3Quat Tobjw;
                        float Maxdist_x = 0;
                        float Maxdist_y = 0;
                        float Maxdist_z = 0;
                        if(AssOBJ->mvpMapPoints.size() > 20)
                        {   
                            checkMPs = true;
                            if(AssOBJ->mbFirstInit)
                            {
                                Tobjw = AssOBJ->mTobjw;
                                Maxdist_x = AssOBJ->mfLength;
                                Maxdist_y = AssOBJ->mfLength;
                                Maxdist_z = AssOBJ->mfLength;
                            }
                            else
                            {   //more accurate
                                Tobjw = AssOBJ->mShape.mTobjw;
                                Maxdist_x = AssOBJ->mShape.a1;
                                Maxdist_y = AssOBJ->mShape.a2;
                                Maxdist_z = AssOBJ->mShape.a3;
                            }
                        }

                        //associate ObjectMap and MapPoints
                        for(size_t j=0;j<obj.mvpMapPoints.size();j++)
                        {   
                            MapPoint* pMP = obj.mvpMapPoints[j];
                            if(pMP->isBad())
                                continue;

                            if(checkMPs)
                            {
                                // check position
                                Eigen::Vector3d ObjPos = Tobjw * Converter::toVector3d(pMP->GetWorldPos());
                                /* float dist = sqrt(ObjPos(0) * ObjPos(0) + ObjPos(1) * ObjPos(1) + ObjPos(2) * ObjPos(2));
                                if(dist > 1.1 * Maxdist)
                                    continue; */
                                if(abs(ObjPos(0)) > AddMPsDistMultiple * Maxdist_x || abs(ObjPos(1)) > AddMPsDistMultiple * Maxdist_y || abs(ObjPos(2)) > AddMPsDistMultiple * Maxdist_z)
                                    continue;

                            }
                            //new MapPoint
                            AssOBJ->AddNewMapPoints(pMP);
                        }

                        AssOBJ->UpdateMapPoints();
                        pFrame->mvObjectMap[i] = AssOBJ;
                        //cout <<"FrameId: "<<pFrame->mnId <<" ObjId: "<< AssOBJ->mnId<<endl;
                        AssOBJ->AddToCloud(obj.mCloud, Twc);
                        mvNewOrChangedObj.push_back(AssOBJ);
                    }
                    else
                    {
                        //creat new ObjectMap
                        int classId = obj.mnClass;
                        float* priorDensity = static_cast<float*>(NULL);
                        pcl::PointCloud<pcl::PointXYZ>::Ptr priorModel = static_cast<pcl::PointCloud<pcl::PointXYZ>::Ptr>(NULL);
                        if (find(mvAvailableClasses.begin(), mvAvailableClasses.end(), classId) == mvAvailableClasses.end())
                        {
                            classId = 0;
                        }
                        else
                        {
                            priorDensity = mmObjectDensities[classId];
                            priorModel = mmObjectModels[classId];
                        }

                        Object_Map* newObjMap = new Object_Map(mpMap, mmObjectConfigs[classId], priorDensity, priorModel, tTest);
                        newObjMap->mnCreatFrameId = pFrame->mnId;
                        newObjMap->mnlatestObsFrameId = pFrame->mnId;
                        newObjMap->mnObs++;
                        newObjMap->mnClass = obj.mnClass;
                        newObjMap->mLastBbox = obj.mBbox;
                        newObjMap->mLastLastBbox = obj.mBbox;
                        newObjMap->mlatestFrameLines = obj.mLines;
                        newObjMap->mvHistoryPos.push_back(obj.mPosMean);
                        newObjMap->AddToCloud(obj.mCloud, Twc);

                        //associate ObjectMap and MapPoints
                        for(size_t j=0;j<obj.mvpMapPoints.size();j++)
                        {   
                            MapPoint* pMP = obj.mvpMapPoints[j];
                            if(pMP->isBad())
                                continue;
                            //new MapPoint
                            newObjMap->AddNewMapPoints(pMP);
                        }
                        
                        // Calculate the mean and standard deviation
                        newObjMap->UpdateMapPoints();

                        pFrame->mvObjectMap[i] = newObjMap;
                        mvNewOrChangedObj.push_back(newObjMap);
                        mpMap->AddObjectMap(newObjMap);
                        
                    }
                }
                
            }

            //End of association, update object_map
            auto startObjectMapTime = std::chrono::system_clock::now();
            //Has been initialized and has a new association
            if(mbInitObjectMap && !mvNewOrChangedObj.empty())
            {   
                
                //6. update ObjectMap
                for(size_t i=0;i < mvNewOrChangedObj.size();i++)
                {
                    Object_Map* pObj = mvNewOrChangedObj[i];
                    //step1. Filter outlier
                    pObj->FilterOutlier(*pFrame);
                    pObj->EIFFilterOutlier();
                    
                    //step2. Calculate (pos) MeanAndStandard
                    pObj->CalculateMeanAndStandard();
                    pObj->CalculatePosMeanAndStandard();

                    //step3. Calculate Translation and Rotation
                    pObj->CalculateObjectPose(*pFrame);

                    //step4. update covisibility relationship
                    pObj->UpdateCovRelation(mvNewOrChangedObj);

                    //step5. History BBox
                    pObj->InsertHistoryBboxAndTwc(*pFrame);

                }
            }
            auto endObjectTime = std::chrono::system_clock::now();
            //cout<<"ObjectMapTime: "<<std::chrono::duration_cast<chrono::milliseconds>(endObjectTime - startObjectMapTime).count()<<endl;
            //cout<<"ObjectTime: "<<std::chrono::duration_cast<chrono::milliseconds>(endObjectTime - startTime).count()<<endl;
            //cout <<"--------------------------------------------------------" <<endl;
        }


        // to push to the Nerf manager after the keyframe is optimized
        if(!mvNewOrChangedObj.empty() && pKF != NULL)
            mpLocalMapper->InsertKeyFrameAndImg(pKF, imColor, imgInstance, imgDepth);

        //update last frame
        mpLastFrame = pFrame;
        mnFramesPassed++;

        auto task_end = std::chrono::high_resolution_clock::now();
        auto task_duration = std::chrono::duration_cast<std::chrono::milliseconds>(task_end - task_start);
        // cout << "Task duration: " << task_duration.count() << " ms" << std::endl;
    }
}

void ObjectManager::ClassPointcloudsFromDepth(const cv::Mat& depth, 
                                              const cv::Mat& imgInstance, 
                                              const vector<Object_Frame>& objectFrames, 
                                              vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& vClassClouds,
                                              const float maxDepth)
{
    // for each object frame
    for (const Object_Frame& objectFrame : objectFrames)
    {
        const uint8_t classId = uint8_t(objectFrame.mBbox.mnClass);
        const Bbox& bbox = objectFrame.mBbox;
        pcl::PointCloud<pcl::PointXYZ>::Ptr objectCloud(new pcl::PointCloud<pcl::PointXYZ>);

        // Iterate through each pixel - row major
        for (int v = bbox.y; v < bbox.y + bbox.height; ++v)
        {
            for (int u = bbox.x; u < bbox.x + bbox.width; ++u)
            {
                // Instance ID check
                if (imgInstance.at<uint8_t>(v,u) != classId)
                    continue;

                // Depth check
                const float z = depth.at<float>(v, u);
                if (z <= 0.0f || std::isnan(z) || z > maxDepth)
                    continue;

                // Back-project the pixel to 3D using intrinsic matrix mK
                Eigen::Vector3f worldPoint(u, v, 1.0f);
                worldPoint = z * mEigenInvK * worldPoint;
                objectCloud->emplace_back(worldPoint.x(), worldPoint.y(), worldPoint.z());
            }
        }

        // Add the object cloud to the class clouds
        // std::cout << "Object cloud size: " << objectCloud->size() << std::endl;
        vClassClouds.push_back(objectCloud);
    }
}

bool ObjectManager::InitObjectMap(Frame* pFrame)
{

    int nObjFrame = pFrame->mvObjectFrame.size();
    pFrame->mvObjectMap = vector<Object_Map*>(nObjFrame,static_cast<Object_Map*>(NULL));

    vector<Object_Frame>& ObjFrame = pFrame->mvObjectFrame;
    const Eigen::Matrix4f Twc = Converter::toMatrix4f(pFrame->mTcw).inverse();

    for(int i=0;i<nObjFrame;i++)
    {
        if(ObjFrame[i].mbBad)
            continue;
        //The mappoints required for initialization are doubled
        if(ObjFrame[i].mvpMapPoints.size() < 10)
        {
            ObjFrame[i].mbBad = true;
            continue;
        }
        
        //create new ObjectMap
        int classId = ObjFrame[i].mnClass;
        float* priorDensity = static_cast<float*>(NULL);
        pcl::PointCloud<pcl::PointXYZ>::Ptr priorModel = static_cast<pcl::PointCloud<pcl::PointXYZ>::Ptr>(NULL);
        if (find(mvAvailableClasses.begin(), mvAvailableClasses.end(), classId) == mvAvailableClasses.end())
        {
            classId = 0;
        }
        else
        {
            priorDensity = mmObjectDensities[classId];
            priorModel = mmObjectModels[classId];
        }
        Object_Map* newObjMap = new Object_Map(mpMap, mmObjectConfigs[classId], priorDensity, priorModel, tTest);
        newObjMap->mnCreatFrameId = pFrame->mnId;
        newObjMap->mnlatestObsFrameId = pFrame->mnId;
        newObjMap->mnObs++;
        newObjMap->mnClass = ObjFrame[i].mnClass;
        newObjMap->mLastBbox = ObjFrame[i].mBbox;
        newObjMap->mLastLastBbox = ObjFrame[i].mBbox;
        newObjMap->mlatestFrameLines = ObjFrame[i].mLines;
        newObjMap->mvHistoryPos.push_back(ObjFrame[i].mPosMean);
        newObjMap->AddToCloud(ObjFrame[i].mCloud, Twc);
        
        //associate ObjectMap and MapPoints
        for(size_t j=0;j<ObjFrame[i].mvpMapPoints.size();j++)
        {
            if(ObjFrame[i].mvpMapPoints[j]->isBad())
                continue;
            MapPoint* pMP = ObjFrame[i].mvpMapPoints[j];

            //new MapPoint
            newObjMap->AddNewMapPoints(pMP);
        }

        // Calculate the mean and standard deviation
        newObjMap->UpdateMapPoints();

        pFrame->mvObjectMap[i] = newObjMap;
        mvNewOrChangedObj.push_back(newObjMap);
        mpMap->AddObjectMap(newObjMap);
        
    }

    if(!mvNewOrChangedObj.empty())
    {
        cout<< "Init Object Map successful"  <<endl;
        return true;
    }  
    else 
        return false;

}

void ObjectManager::SetLocalMapper(LocalMapping* pLocalMapper)
{
    mpLocalMapper = pLocalMapper;
}

} // namespace ORB_SLAM2