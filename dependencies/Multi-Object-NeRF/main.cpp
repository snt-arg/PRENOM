/** @file   main.cpp
 *  @author Xiao Han
 *  @date 2022.11.8
 * 
 * @modified by Saad Ejaz
 */

#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <unistd.h>
#include <vector>
#include <thread>
#include <Eigen/Core>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>

#include "nerf_manager.h"
#include <pangolin/pangolin.h>

#include "third_party/tiny-cuda-nn/dependencies/json/json.hpp"

using json = nlohmann::json;
using namespace std;

// Visualization class
class viewer
{
public:
    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    viewer(){};
    void GenerateRays();
    void Run();
    void DrawObjects(std::shared_ptr<nerf::NeRF> pNeRF, bool showBbox, bool showMesh);
    void DrawFrameAndRays(bool showframe, bool showObs);

    float ViewpointX = 0;
    float ViewpointY = -0.7;
    float ViewpointZ = -1.8;
    float ViewpointF = 500;

    vector<Eigen::Matrix4f> AllTwc;
    vector<vector<Eigen::Vector3f>> Rays;
    vector<vector<int>> RaysCls;
    vector<std::shared_ptr<nerf::NeRF>> vpNeRFs;
    float fx;
    float fy;
    float cx;
    float cy;
};

void viewer::GenerateRays()
{
    Rays.resize(AllTwc.size());
    RaysCls.resize(AllTwc.size());

    for(int i=0;i<vpNeRFs.size();i++)
    {
        vector<nerf::FrameIdAndBbox> frameIdBboxs = vpNeRFs[i]->GetFrameIdAndBBox();
        for(auto& frameIdBbox : frameIdBboxs)
        {
            uint32_t boxx = frameIdBbox.x;
            uint32_t boxy = frameIdBbox.y;
            uint32_t boxw = frameIdBbox.w;
            uint32_t boxh = frameIdBbox.h;

            uint32_t x = boxx + boxw / 2;
            uint32_t y = boxy + boxh / 2;
            Eigen::Vector3f d= Eigen::Vector3f((float(x) - cx) / fx,(float(y) - cy) / fy,1.0f);
            
            Rays[frameIdBbox.FrameId].push_back(d.normalized());
            RaysCls[frameIdBbox.FrameId].push_back(i);
        }
    }

}

void viewer::Run()
{   

    pangolin::CreateWindowAndBind("object pose viewer",1024,768);
    if(glewInit() != GLEW_OK)
        throw std::runtime_error("glewInit failed");
    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


    //Panel
    pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuShowFrames("menu.Show Frames",true,true);
    pangolin::Var<bool> menuShowBbox("menu.Show Bbox",true,true);
    pangolin::Var<bool> menuShowMesh("menu.Show Mesh",true,true);
    pangolin::Var<bool> menuShowObs("menu.Show Obs",false,true);


    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024,768,ViewpointF,ViewpointF,512,389,0.1,1000),
                pangolin::ModelViewLookAt(ViewpointX,ViewpointY,ViewpointZ, 0,0,0,0.0,-1.0, 0.0)
                );

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));


    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();
    s_cam.Follow(Twc);

    /* cout <<glGetString(GL_VENDOR)<<endl;
    cout <<glGetString(GL_RENDERER)<<endl;
    cout <<glGetString(GL_VERSION)<<endl; */

    while(1)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f,1.0f,1.0f,1.0f);
        //ground XYZ
        glBegin ( GL_LINES );
        glColor3f ( 1.0f,0.f,0.f );
        glVertex3f( 0,0,0 );
        glVertex3f( 1,0,0 );
        glColor3f( 0.f,1.0f,0.f);
        glVertex3f( 0,0,0 );
        glVertex3f( 0,1,0 );
        glColor3f( 0.f,0.f,1.f);
        glVertex3f( 0,0,0 );
        glVertex3f( 0,0,1 );
        glEnd();

        DrawFrameAndRays(menuShowFrames,menuShowObs);
        for(int i=0;i<vpNeRFs.size();i++)
        {
            DrawObjects(vpNeRFs[i],menuShowBbox,menuShowMesh);
        }
        
        pangolin::FinishFrame();
        usleep(3000);
    }
}

void viewer::DrawObjects(std::shared_ptr<nerf::NeRF> pNeRF, bool showBbox, bool showMesh)
{   

    glPushMatrix();
    Eigen::Matrix4f pose = pNeRF->GetObjTow().inverse();
    glMultMatrixf(pose.data());
    nerf::BoundingBox Bbox = pNeRF->GetBoundingBox();
    const float a1 = Bbox.min[0];
    const float a2 = Bbox.min[1];
    const float a3 = Bbox.min[2];
    const float a4 = Bbox.max[0];
    const float a5 = Bbox.max[1];
    const float a6 = Bbox.max[2];
    const float linewidth = 1;
    
    if(showBbox)
    {
        glLineWidth(linewidth);
        glBegin(GL_LINES);
        glColor3f(1.0f, 0.0f, 0.0f);
        //1
        glVertex3f(a4, a5, a6);
        glVertex3f(a1, a5, a6);
        //2
        glVertex3f(a1, a5, a6);
        glVertex3f(a1, a2, a6);
        //3
        glVertex3f(a1, a2, a6);
        glVertex3f(a4, a2, a6);
        //4
        glVertex3f(a4, a2, a6);
        glVertex3f(a4, a5, a6);
        //5
        glVertex3f(a4, a5, a3);
        glVertex3f(a1, a5, a3);
        //6
        glVertex3f(a1, a5, a3);
        glVertex3f(a1, a2, a3);
        //7
        glVertex3f(a1, a2, a3);
        glVertex3f(a4, a2, a3);
        //8
        glVertex3f(a4, a2, a3);
        glVertex3f(a4, a5, a3);
        //1
        glVertex3f(a4, a5, a6);
        glVertex3f(a4, a5, a3);
        //2
        glVertex3f(a1, a5, a6);
        glVertex3f(a1, a5, a3);
        //3
        glVertex3f(a1, a2, a6);
        glVertex3f(a1, a2, a3);
        //4
        glVertex3f(a4, a2, a6);
        glVertex3f(a4, a2, a3);
        glEnd();

        //XYZ Coordinate
        float w = 0.1;
        glLineWidth(linewidth);
        glBegin ( GL_LINES );
        glColor3f ( 1.0f,0.f,0.f );
        glVertex3f( 0,0,0 );
        glVertex3f( w,0,0 );
        glColor3f( 0.f,1.0f,0.f);
        glVertex3f( 0,0,0 );
        glVertex3f( 0,w,0 );
        glColor3f( 0.f,0.f,1.f);
        glVertex3f( 0,0,0 );
        glVertex3f( 0,0,w );
        glEnd();
    }

    if(showMesh)
        pNeRF->DrawCPUMesh();
    //pNeRF->DrawMesh();
    
    glPopMatrix();
}


void viewer::DrawFrameAndRays(bool showframe, bool showObs)
{
    const float w = 0.02;
    const float h = w*0.75;
    const float z = w*0.6;
    
    // color.
    vector<vector<float>> colors = { {0.9f,0.1f,0.1f},
                                     {0.1f,0.1f,0.9f},
                                     {0.1f,0.9f,0.1f},
                                     {0.9f,0.05f,0.9f}};

    if(showframe)
    {
        for(size_t i=0; i<AllTwc.size(); i++)
        {   
            if(!Rays[i].empty())
            {
                Eigen::Matrix4f pose = AllTwc[i];

                glPushMatrix();
                glMultMatrixf(pose.data());
                glLineWidth(1.0);
                glColor3f(0.0f,0.0f,1.0f);
                glBegin(GL_LINES);
                glVertex3f(0,0,0);
                glVertex3f(w,h,z);
                glVertex3f(0,0,0);
                glVertex3f(w,-h,z);
                glVertex3f(0,0,0);
                glVertex3f(-w,-h,z);
                glVertex3f(0,0,0);
                glVertex3f(-w,h,z);
                glVertex3f(w,h,z);
                glVertex3f(w,-h,z);
                glVertex3f(-w,h,z);
                glVertex3f(-w,-h,z);
                glVertex3f(-w,h,z);
                glVertex3f(w,h,z);
                glVertex3f(-w,-h,z);
                glVertex3f(w,-h,z);
                glEnd();
                
                if(showObs)
                {
                    for(int j=0;j<Rays[i].size();j++)
                    {   
                        Eigen::Vector3f ray = Rays[i][j];
                        vector<float> color = colors[RaysCls[i][j]];
                        glColor3f(color[0],color[1],color[2]);
                        glBegin(GL_LINES);
                        glVertex3f(0,0,0);
                        glVertex3f(ray[0],ray[1],ray[2]);
                        glEnd();
                    }
                }
                glPopMatrix();
            }
        }
    }

}

int main(int argc, char** argv)
{

    cout<<"......Multi-Object NeRF Offline......"<<endl;
    if(argc != 4)
    {
        cerr << "param error..."<<endl<<"./build/OfflineNeRF nerf_config_path system_config_path dataset_path"<<endl;
        return 0;
    }

    // defining them here for readability
    const string nerfConfigPath = string(argv[1]);
    const string systemConfigPath = string(argv[2]);
    const string datasetPath = string(argv[3]);
    
    std::ifstream file(systemConfigPath);
    if(!file)
    {
        cerr << "config file error..."<<endl;
        return false;
    }
	json systemConfig = json::parse(file, nullptr, true, true);
    
    const bool UseGTdepth = systemConfig["use_depth"];
    const vector<int> object_indices = systemConfig["object_indices"];

    //Only synthetic datasets can be used
    string objPath = datasetPath + "/obj_offline";
    if(access(objPath.c_str(),0) != 0)
    {
        cerr << "Only the synthetic dataset can be used!" <<endl;
        return 0;
    }   

    //Read training information for each object
    const int num_objects = object_indices.size();
    vector<string> vObjPath(num_objects);
    for(int i=0; i<num_objects; i++)
        vObjPath[i] = objPath + "/" + to_string(object_indices[i]) + ".txt";

    //initialization
    nerf::NerfManagerOffline nerfManager(datasetPath, nerfConfigPath, UseGTdepth);
    nerfManager.Init();
    nerfManager.ReadDataset();

    //Create NeRF models and Training
    for(int i=0;i<vObjPath.size();i++)
        nerfManager.CreateNeRF(vObjPath[i], systemConfig);
   
    //visualization
    if (systemConfig["visualize"])
    {
        std::shared_ptr<viewer> view = std::make_shared<viewer>();
        view->AllTwc = nerfManager.GetAllTwc();
        nerfManager.GetIntrinsics(view->fx,view->fy,view->cx,view->cy);
        view->vpNeRFs = nerfManager.GetAllNeRF();
        view->GenerateRays();
        view->Run();
    }
    
    nerfManager.WaitThreadsEnd();

    return 0;
}


