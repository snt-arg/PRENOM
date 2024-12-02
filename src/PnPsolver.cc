/**
* This file is part of ORB-SLAM2.
* This file is a modified version of EPnP <http://cvlab.epfl.ch/EPnP/index.php>, see FreeBSD license below.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

/**
* Copyright (c) 2009, V. Lepetit, EPFL
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
*    list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* The views and conclusions contained in the software and documentation are those
* of the authors and should not be interpreted as representing official policies,
*   either expressed or implied, of the FreeBSD Project
*/

#include <iostream>

#include "PnPsolver.h"

#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>
#include "dependencies/DBoW2/DUtils/Random.h"
#include <algorithm>

using namespace std;

namespace ORB_SLAM2
{


PnPsolver::PnPsolver(const Frame &F, const vector<MapPoint*> &vpMapPointMatches):
    pws(0), us(0), alphas(0), pcs(0), maximum_number_of_correspondences(0), number_of_correspondences(0), mnInliersi(0),
    mnIterations(0), mnBestInliers(0), N(0)
{
    mvpMapPointMatches = vpMapPointMatches;
    mvP2D.reserve(F.mvpMapPoints.size());
    mvSigma2.reserve(F.mvpMapPoints.size());
    mvP3Dw.reserve(F.mvpMapPoints.size());
    mvKeyPointIndices.reserve(F.mvpMapPoints.size());
    mvAllIndices.reserve(F.mvpMapPoints.size());

    int idx=0;
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];

        if(pMP)
        {
            if(!pMP->isBad())
            {
                const cv::KeyPoint &kp = F.mvKeysUn[i];

                mvP2D.push_back(kp.pt);
                mvSigma2.push_back(F.mvLevelSigma2[kp.octave]);

                cv::Mat Pos = pMP->GetWorldPos();
                mvP3Dw.push_back(cv::Point3f(Pos.at<float>(0),Pos.at<float>(1), Pos.at<float>(2)));

                mvKeyPointIndices.push_back(i);
                mvAllIndices.push_back(idx);               

                idx++;
            }
        }
    }

    // Set camera calibration parameters
    fu = F.fx;
    fv = F.fy;
    uc = F.cx;
    vc = F.cy;

    SetRansacParameters();
}

PnPsolver::~PnPsolver()
{
  delete [] pws;
  delete [] us;
  delete [] alphas;
  delete [] pcs;
}


void PnPsolver::SetRansacParameters(double probability, int minInliers, int maxIterations, int minSet, float epsilon, float th2)
{
    mRansacProb = probability;
    mRansacMinInliers = minInliers;
    mRansacMaxIts = maxIterations;
    mRansacEpsilon = epsilon;
    mRansacMinSet = minSet;

    N = mvP2D.size(); // number of correspondences

    mvbInliersi.resize(N);

    // Adjust Parameters according to number of correspondences
    int nMinInliers = N*mRansacEpsilon;
    if(nMinInliers<mRansacMinInliers)
        nMinInliers=mRansacMinInliers;
    if(nMinInliers<minSet)
        nMinInliers=minSet;
    mRansacMinInliers = nMinInliers;

    if(mRansacEpsilon<(float)mRansacMinInliers/N)
        mRansacEpsilon=(float)mRansacMinInliers/N;

    // Set RANSAC iterations according to probability, epsilon, and max iterations
    int nIterations;

    if(mRansacMinInliers==N)
        nIterations=1;
    else
        nIterations = ceil(log(1-mRansacProb)/log(1-pow(mRansacEpsilon,3)));

    mRansacMaxIts = max(1,min(nIterations,mRansacMaxIts));

    mvMaxError.resize(mvSigma2.size());
    for(size_t i=0; i<mvSigma2.size(); i++)
        mvMaxError[i] = mvSigma2[i]*th2;
}

cv::Mat PnPsolver::find(vector<bool> &vbInliers, int &nInliers)
{
    bool bFlag;
    return iterate(mRansacMaxIts,bFlag,vbInliers,nInliers);    
}

cv::Mat PnPsolver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers)
{
    bNoMore = false;
    vbInliers.clear();
    nInliers=0;

    set_maximum_number_of_correspondences(mRansacMinSet);

    if(N<mRansacMinInliers)
    {
        bNoMore = true;
        return cv::Mat();
    }

    vector<size_t> vAvailableIndices;

    int nCurrentIterations = 0;
    while(mnIterations<mRansacMaxIts || nCurrentIterations<nIterations)
    {
        nCurrentIterations++;
        mnIterations++;
        reset_correspondences();

        vAvailableIndices = mvAllIndices;

        // Get min set of points
        for(short i = 0; i < mRansacMinSet; ++i)
        {
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size()-1);

            int idx = vAvailableIndices[randi];

            add_correspondence(mvP3Dw[idx].x,mvP3Dw[idx].y,mvP3Dw[idx].z,mvP2D[idx].x,mvP2D[idx].y);

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }

        // Compute camera pose
        compute_pose(mRi, mti);

        // Check inliers
        CheckInliers();

        if(mnInliersi>=mRansacMinInliers)
        {
            // If it is the best solution so far, save it
            if(mnInliersi>mnBestInliers)
            {
                mvbBestInliers = mvbInliersi;
                mnBestInliers = mnInliersi;

                cv::Mat Rcw(3,3,CV_64F,mRi);
                cv::Mat tcw(3,1,CV_64F,mti);
                Rcw.convertTo(Rcw,CV_32F);
                tcw.convertTo(tcw,CV_32F);
                mBestTcw = cv::Mat::eye(4,4,CV_32F);
                Rcw.copyTo(mBestTcw.rowRange(0,3).colRange(0,3));
                tcw.copyTo(mBestTcw.rowRange(0,3).col(3));
            }

            if(Refine())
            {
                nInliers = mnRefinedInliers;
                vbInliers = vector<bool>(mvpMapPointMatches.size(),false);
                for(int i=0; i<N; i++)
                {
                    if(mvbRefinedInliers[i])
                        vbInliers[mvKeyPointIndices[i]] = true;
                }
                return mRefinedTcw.clone();
            }

        }
    }

    if(mnIterations>=mRansacMaxIts)
    {
        bNoMore=true;
        if(mnBestInliers>=mRansacMinInliers)
        {
            nInliers=mnBestInliers;
            vbInliers = vector<bool>(mvpMapPointMatches.size(),false);
            for(int i=0; i<N; i++)
            {
                if(mvbBestInliers[i])
                    vbInliers[mvKeyPointIndices[i]] = true;
            }
            return mBestTcw.clone();
        }
    }

    return cv::Mat();
}

bool PnPsolver::Refine()
{
    vector<int> vIndices;
    vIndices.reserve(mvbBestInliers.size());

    for(size_t i=0; i<mvbBestInliers.size(); i++)
    {
        if(mvbBestInliers[i])
        {
            vIndices.push_back(i);
        }
    }

    set_maximum_number_of_correspondences(vIndices.size());

    reset_correspondences();

    for(size_t i=0; i<vIndices.size(); i++)
    {
        int idx = vIndices[i];
        add_correspondence(mvP3Dw[idx].x,mvP3Dw[idx].y,mvP3Dw[idx].z,mvP2D[idx].x,mvP2D[idx].y);
    }

    // Compute camera pose
    compute_pose(mRi, mti);

    // Check inliers
    CheckInliers();

    mnRefinedInliers =mnInliersi;
    mvbRefinedInliers = mvbInliersi;

    if(mnInliersi>mRansacMinInliers)
    {
        cv::Mat Rcw(3,3,CV_64F,mRi);
        cv::Mat tcw(3,1,CV_64F,mti);
        Rcw.convertTo(Rcw,CV_32F);
        tcw.convertTo(tcw,CV_32F);
        mRefinedTcw = cv::Mat::eye(4,4,CV_32F);
        Rcw.copyTo(mRefinedTcw.rowRange(0,3).colRange(0,3));
        tcw.copyTo(mRefinedTcw.rowRange(0,3).col(3));
        return true;
    }

    return false;
}


void PnPsolver::CheckInliers()
{
    mnInliersi=0;

    for(int i=0; i<N; i++)
    {
        cv::Point3f P3Dw = mvP3Dw[i];
        cv::Point2f P2D = mvP2D[i];

        float Xc = mRi[0][0]*P3Dw.x+mRi[0][1]*P3Dw.y+mRi[0][2]*P3Dw.z+mti[0];
        float Yc = mRi[1][0]*P3Dw.x+mRi[1][1]*P3Dw.y+mRi[1][2]*P3Dw.z+mti[1];
        float invZc = 1/(mRi[2][0]*P3Dw.x+mRi[2][1]*P3Dw.y+mRi[2][2]*P3Dw.z+mti[2]);

        double ue = uc + fu * Xc * invZc;
        double ve = vc + fv * Yc * invZc;

        float distX = P2D.x-ue;
        float distY = P2D.y-ve;

        float error2 = distX*distX+distY*distY;

        if(error2<mvMaxError[i])
        {
            mvbInliersi[i]=true;
            mnInliersi++;
        }
        else
        {
            mvbInliersi[i]=false;
        }
    }
}


void PnPsolver::set_maximum_number_of_correspondences(int n)
{
  if (maximum_number_of_correspondences < n) {
    if (pws != 0) delete [] pws;
    if (us != 0) delete [] us;
    if (alphas != 0) delete [] alphas;
    if (pcs != 0) delete [] pcs;

    maximum_number_of_correspondences = n;
    pws = new double[3 * maximum_number_of_correspondences];
    us = new double[2 * maximum_number_of_correspondences];
    alphas = new double[4 * maximum_number_of_correspondences];
    pcs = new double[3 * maximum_number_of_correspondences];
  }
}

void PnPsolver::reset_correspondences(void)
{
  number_of_correspondences = 0;
}

void PnPsolver::add_correspondence(double X, double Y, double Z, double u, double v)
{
  pws[3 * number_of_correspondences    ] = X;
  pws[3 * number_of_correspondences + 1] = Y;
  pws[3 * number_of_correspondences + 2] = Z;

  us[2 * number_of_correspondences    ] = u;
  us[2 * number_of_correspondences + 1] = v;

  number_of_correspondences++;
}

void PnPsolver::choose_control_points(void)
{
    // Take C0 as the reference points centroid:
    cws[0][0] = cws[0][1] = cws[0][2] = 0;
    for (int i = 0; i < number_of_correspondences; i++)
        for (int j = 0; j < 3; j++)
            cws[0][j] += pws[3 * i + j];

    for (int j = 0; j < 3; j++)
        cws[0][j] /= number_of_correspondences;

    // Take C1, C2, and C3 from PCA on the reference points:
    cv::Mat PW0 = cv::Mat(number_of_correspondences, 3, CV_64F);
    for (int i = 0; i < number_of_correspondences; i++)
        for (int j = 0; j < 3; j++)
            PW0.at<double>(i, j) = pws[3 * i + j] - cws[0][j];

    // Compute covariance matrix PW0tPW0
    cv::Mat PW0tPW0 = PW0.t() * PW0;

    // Perform SVD
    cv::Mat eigenvalues, DC, UCt;
    cv::SVD::compute(PW0tPW0, eigenvalues, DC, UCt);

    // Compute the control points
    for (int i = 1; i < 4; i++) {
        double k = sqrt(DC.at<double>(i - 1) / number_of_correspondences);
        for (int j = 0; j < 3; j++)
            cws[i][j] = cws[0][j] + k * UCt.at<double>(i - 1, j);
    }
}


void PnPsolver::compute_barycentric_coordinates(void)
{
    double cc[3 * 3];
    cv::Mat CC = cv::Mat(3, 3, CV_64F, cc);

    // Fill the CC matrix with control points
    for (int i = 0; i < 3; i++) {
        for (int j = 1; j < 4; j++) {
            CC.at<double>(i, j - 1) = cws[j][i] - cws[0][i];
        }
    }

    // Invert the CC matrix
    cv::Mat CC_inv = CC.inv(cv::DECOMP_SVD);

    for (int i = 0; i < number_of_correspondences; i++) {
        double* pi = pws + 3 * i; // Pointer to current point
        double* a = alphas + 4 * i; // Pointer to barycentric coordinates

        for (int j = 0; j < 3; j++) {
            a[1 + j] =
                CC_inv.at<double>(0, j) * (pi[0] - cws[0][0]) +
                CC_inv.at<double>(1, j) * (pi[1] - cws[0][1]) +
                CC_inv.at<double>(2, j) * (pi[2] - cws[0][2]);
        }
        a[0] = 1.0 - a[1] - a[2] - a[3];
    }
}

void PnPsolver::fill_M(cv::Mat* M, const int row, const double* as, const double u, const double v)
{
    int M1_row = row;
    int M2_row = row + 1;

    for (int i = 0; i < 4; i++) {
        M->at<double>(M1_row, 3 * i) = as[i] * fu;
        M->at<double>(M1_row, 3 * i + 1) = 0.0;
        M->at<double>(M1_row, 3 * i + 2) = as[i] * (uc - u);

        M->at<double>(M2_row, 3 * i) = 0.0;
        M->at<double>(M2_row, 3 * i + 1) = as[i] * fv;
        M->at<double>(M2_row, 3 * i + 2) = as[i] * (vc - v);
    }
}

void PnPsolver::compute_ccs(const double * betas, const double * ut)
{
  for(int i = 0; i < 4; i++)
    ccs[i][0] = ccs[i][1] = ccs[i][2] = 0.0f;

  for(int i = 0; i < 4; i++) {
    const double * v = ut + 12 * (11 - i);
    for(int j = 0; j < 4; j++)
      for(int k = 0; k < 3; k++)
	ccs[j][k] += betas[i] * v[3 * j + k];
  }
}

void PnPsolver::compute_pcs(void)
{
  for(int i = 0; i < number_of_correspondences; i++) {
    double * a = alphas + 4 * i;
    double * pc = pcs + 3 * i;

    for(int j = 0; j < 3; j++)
      pc[j] = a[0] * ccs[0][j] + a[1] * ccs[1][j] + a[2] * ccs[2][j] + a[3] * ccs[3][j];
  }
}

double PnPsolver::compute_pose(double R[3][3], double t[3])
{
    // Step 1: Choose control points and compute barycentric coordinates
    choose_control_points();
    compute_barycentric_coordinates();

    // Step 2: Construct M matrix
    cv::Mat M = cv::Mat::zeros(2 * number_of_correspondences, 12, CV_64F);
    for (int i = 0; i < number_of_correspondences; i++) {
        fill_M(&M, 2 * i, alphas + 4 * i, us[2 * i], us[2 * i + 1]);
    }

    // Step 3: Compute MtM = M^T * M
    cv::Mat MtM = M.t() * M;

    // Step 4: Perform SVD on MtM
    cv::Mat eigenvalues, U, Vt;
    cv::SVD::compute(MtM, eigenvalues, U, Vt);

    // Step 5: Construct L_6x10 and Rho matrices
    cv::Mat L_6x10 = cv::Mat::zeros(6, 10, CV_64F);
    cv::Mat Rho = cv::Mat::zeros(6, 1, CV_64F);

    compute_L_6x10(Vt, L_6x10); // Use V^T for computation
    compute_rho(Rho.ptr<double>());

    double Betas[4][4], rep_errors[4];
    double Rs[4][3][3], ts[4][3];

    // Step 6: Find and refine betas using Gauss-Newton
    find_betas_approx_1(L_6x10, Rho, Betas[1]);
    gauss_newton(L_6x10, Rho, Betas[1]);
    rep_errors[1] = compute_R_and_t(Vt.ptr<double>(), Betas[1], Rs[1], ts[1]);

    find_betas_approx_2(L_6x10, Rho, Betas[2]);
    gauss_newton(L_6x10, Rho, Betas[2]);
    rep_errors[2] = compute_R_and_t(Vt.ptr<double>(), Betas[2], Rs[2], ts[2]);

    find_betas_approx_3(L_6x10, Rho, Betas[3]);
    gauss_newton(L_6x10, Rho, Betas[3]);
    rep_errors[3] = compute_R_and_t(Vt.ptr<double>(), Betas[3], Rs[3], ts[3]);

    // Step 7: Find the best solution with the lowest reprojection error
    int N = 1;
    if (rep_errors[2] < rep_errors[1]) N = 2;
    if (rep_errors[3] < rep_errors[N]) N = 3;

    copy_R_and_t(Rs[N], ts[N], R, t);

    return rep_errors[N];
}

void PnPsolver::copy_R_and_t(const double R_src[3][3], const double t_src[3],
			double R_dst[3][3], double t_dst[3])
{
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      R_dst[i][j] = R_src[i][j];
    t_dst[i] = t_src[i];
  }
}

double PnPsolver::dist2(const double * p1, const double * p2)
{
  return
    (p1[0] - p2[0]) * (p1[0] - p2[0]) +
    (p1[1] - p2[1]) * (p1[1] - p2[1]) +
    (p1[2] - p2[2]) * (p1[2] - p2[2]);
}

double PnPsolver::dot(const double * v1, const double * v2)
{
  return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

double PnPsolver::reprojection_error(const double R[3][3], const double t[3])
{
  double sum2 = 0.0;

  for(int i = 0; i < number_of_correspondences; i++) {
    double * pw = pws + 3 * i;
    double Xc = dot(R[0], pw) + t[0];
    double Yc = dot(R[1], pw) + t[1];
    double inv_Zc = 1.0 / (dot(R[2], pw) + t[2]);
    double ue = uc + fu * Xc * inv_Zc;
    double ve = vc + fv * Yc * inv_Zc;
    double u = us[2 * i], v = us[2 * i + 1];

    sum2 += sqrt( (u - ue) * (u - ue) + (v - ve) * (v - ve) );
  }

  return sum2 / number_of_correspondences;
}

void PnPsolver::estimate_R_and_t(double R[3][3], double t[3])
{
    double pc0[3] = {0.0, 0.0, 0.0};
    double pw0[3] = {0.0, 0.0, 0.0};

    // Compute centroids of `pcs` and `pws`
    for (int i = 0; i < number_of_correspondences; i++) {
        const double* pc = pcs + 3 * i;
        const double* pw = pws + 3 * i;

        for (int j = 0; j < 3; j++) {
            pc0[j] += pc[j];
            pw0[j] += pw[j];
        }
    }

    for (int j = 0; j < 3; j++) {
        pc0[j] /= number_of_correspondences;
        pw0[j] /= number_of_correspondences;
    }

    // Compute matrix ABt
    cv::Mat ABt = cv::Mat::zeros(3, 3, CV_64F);
    for (int i = 0; i < number_of_correspondences; i++) {
        const double* pc = pcs + 3 * i;
        const double* pw = pws + 3 * i;

        for (int j = 0; j < 3; j++) {
            ABt.at<double>(j, 0) += (pc[j] - pc0[j]) * (pw[0] - pw0[0]);
            ABt.at<double>(j, 1) += (pc[j] - pc0[j]) * (pw[1] - pw0[1]);
            ABt.at<double>(j, 2) += (pc[j] - pc0[j]) * (pw[2] - pw0[2]);
        }
    }

    // Perform SVD on ABt
    cv::Mat W, U, Vt;
    cv::SVD::compute(ABt, W, U, Vt);

    // Compute R = U * Vt
    cv::Mat Rt = U * Vt;
    cv::Mat R_mat = Rt.t(); // Transpose to get R in the correct orientation

    // Copy R_mat to the output R
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            R[i][j] = R_mat.at<double>(i, j);
        }
    }

    // Ensure R has a positive determinant
    const double det = cv::determinant(R_mat);
    if (det < 0) {
        for (int j = 0; j < 3; j++) {
            R[2][j] = -R[2][j];
        }
    }

    // Compute t = pc0 - R * pw0
    cv::Mat pc0_mat = cv::Mat(3, 1, CV_64F, pc0);
    cv::Mat pw0_mat = cv::Mat(3, 1, CV_64F, pw0);
    cv::Mat t_mat = pc0_mat - R_mat * pw0_mat;

    // Copy t_mat to the output t
    for (int i = 0; i < 3; i++) {
        t[i] = t_mat.at<double>(i, 0);
    }
}

void PnPsolver::print_pose(const double R[3][3], const double t[3])
{
  cout << R[0][0] << " " << R[0][1] << " " << R[0][2] << " " << t[0] << endl;
  cout << R[1][0] << " " << R[1][1] << " " << R[1][2] << " " << t[1] << endl;
  cout << R[2][0] << " " << R[2][1] << " " << R[2][2] << " " << t[2] << endl;
}

void PnPsolver::solve_for_sign(void)
{
  if (pcs[2] < 0.0) {
    for(int i = 0; i < 4; i++)
      for(int j = 0; j < 3; j++)
	ccs[i][j] = -ccs[i][j];

    for(int i = 0; i < number_of_correspondences; i++) {
      pcs[3 * i    ] = -pcs[3 * i];
      pcs[3 * i + 1] = -pcs[3 * i + 1];
      pcs[3 * i + 2] = -pcs[3 * i + 2];
    }
  }
}

double PnPsolver::compute_R_and_t(const double * ut, const double * betas,
			     double R[3][3], double t[3])
{
  compute_ccs(betas, ut);
  compute_pcs();

  solve_for_sign();

  estimate_R_and_t(R, t);

  return reprojection_error(R, t);
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_1 = [B11 B12     B13         B14]

void PnPsolver::find_betas_approx_1(const cv::Mat& L_6x10, const cv::Mat& Rho, double* betas)
{
    // Create reduced L_6x4 and solution vector B4
    cv::Mat L_6x4 = cv::Mat(6, 4, CV_64F);
    cv::Mat B4 = cv::Mat(4, 1, CV_64F);

    // Fill L_6x4 from L_6x10
    for (int i = 0; i < 6; i++) {
        L_6x4.at<double>(i, 0) = L_6x10.at<double>(i, 0);
        L_6x4.at<double>(i, 1) = L_6x10.at<double>(i, 1);
        L_6x4.at<double>(i, 2) = L_6x10.at<double>(i, 3);
        L_6x4.at<double>(i, 3) = L_6x10.at<double>(i, 6);
    }

    // Solve the linear system L_6x4 * B4 = Rho
    cv::solve(L_6x4, Rho, B4, cv::DECOMP_SVD);

    // Extract the solution from B4
    double b4[4];
    for (int i = 0; i < 4; i++) {
        b4[i] = B4.at<double>(i, 0);
    }

    // Compute betas based on the sign of b4[0]
    if (b4[0] < 0) {
        betas[0] = sqrt(-b4[0]);
        betas[1] = -b4[1] / betas[0];
        betas[2] = -b4[2] / betas[0];
        betas[3] = -b4[3] / betas[0];
    } else {
        betas[0] = sqrt(b4[0]);
        betas[1] = b4[1] / betas[0];
        betas[2] = b4[2] / betas[0];
        betas[3] = b4[3] / betas[0];
    }
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_2 = [B11 B12 B22                            ]

void PnPsolver::find_betas_approx_2(const cv::Mat& L_6x10, const cv::Mat& Rho, double* betas)
{
    // Create reduced L_6x3 and solution vector B3
    cv::Mat L_6x3 = cv::Mat(6, 3, CV_64F);
    cv::Mat B3 = cv::Mat(3, 1, CV_64F);

    // Fill L_6x3 from L_6x10
    for (int i = 0; i < 6; i++) {
        L_6x3.at<double>(i, 0) = L_6x10.at<double>(i, 0);
        L_6x3.at<double>(i, 1) = L_6x10.at<double>(i, 1);
        L_6x3.at<double>(i, 2) = L_6x10.at<double>(i, 2);
    }

    // Solve the linear system L_6x3 * B3 = Rho
    cv::solve(L_6x3, Rho, B3, cv::DECOMP_SVD);

    // Extract the solution from B3
    double b3[3];
    for (int i = 0; i < 3; i++) {
        b3[i] = B3.at<double>(i, 0);
    }

    // Compute betas based on the signs of b3[0] and b3[2]
    if (b3[0] < 0) {
        betas[0] = sqrt(-b3[0]);
        betas[1] = (b3[2] < 0) ? sqrt(-b3[2]) : 0.0;
    } else {
        betas[0] = sqrt(b3[0]);
        betas[1] = (b3[2] > 0) ? sqrt(b3[2]) : 0.0;
    }

    // Adjust the sign of betas[0] based on b3[1]
    if (b3[1] < 0) {
        betas[0] = -betas[0];
    }

    // Set unused betas to 0
    betas[2] = 0.0;
    betas[3] = 0.0;
}


// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_3 = [B11 B12 B22 B13 B23                    ]

void PnPsolver::find_betas_approx_3(const cv::Mat& L_6x10, const cv::Mat& Rho, double* betas)
{
    // Create reduced L_6x5 and solution vector B5
    cv::Mat L_6x5 = cv::Mat(6, 5, CV_64F);
    cv::Mat B5 = cv::Mat(5, 1, CV_64F);

    // Fill L_6x5 from L_6x10
    for (int i = 0; i < 6; i++) {
        L_6x5.at<double>(i, 0) = L_6x10.at<double>(i, 0);
        L_6x5.at<double>(i, 1) = L_6x10.at<double>(i, 1);
        L_6x5.at<double>(i, 2) = L_6x10.at<double>(i, 2);
        L_6x5.at<double>(i, 3) = L_6x10.at<double>(i, 3);
        L_6x5.at<double>(i, 4) = L_6x10.at<double>(i, 4);
    }

    // Solve the linear system L_6x5 * B5 = Rho
    cv::solve(L_6x5, Rho, B5, cv::DECOMP_SVD);

    // Extract the solution from B5
    double b5[5];
    for (int i = 0; i < 5; i++) {
        b5[i] = B5.at<double>(i, 0);
    }

    // Compute betas based on the signs of b5[0] and b5[2]
    if (b5[0] < 0) {
        betas[0] = sqrt(-b5[0]);
        betas[1] = (b5[2] < 0) ? sqrt(-b5[2]) : 0.0;
    } else {
        betas[0] = sqrt(b5[0]);
        betas[1] = (b5[2] > 0) ? sqrt(b5[2]) : 0.0;
    }

    // Adjust the sign of betas[0] based on b5[1]
    if (b5[1] < 0) {
        betas[0] = -betas[0];
    }

    // Compute betas[2] and set betas[3]
    betas[2] = b5[3] / betas[0];
    betas[3] = 0.0;
}


void PnPsolver::compute_L_6x10(const cv::Mat& ut, cv::Mat& l_6x10)
{
    // Ensure `ut` is a 12x12 matrix
    CV_Assert(ut.rows == 12 && ut.cols == 12 && ut.type() == CV_64F);

    // Ensure `l_6x10` is a 6x10 matrix
    l_6x10 = cv::Mat::zeros(6, 10, CV_64F);

    // Extract the last four rows (v[0], v[1], v[2], v[3]) from `ut`
    std::vector<cv::Mat> v(4);
    for (int i = 0; i < 4; i++) {
        v[i] = ut.row(11 - i).reshape(1, 1); // Extract each row as a 1x12 vector
    }

    // Compute the differences (dv) between pairs of points in v
    cv::Mat dv[4][6]; // dv[i][j] stores a 1x3 difference vector

    for (int i = 0; i < 4; i++) {
        int a = 0, b = 1;
        for (int j = 0; j < 6; j++) {
            dv[i][j] = v[i].colRange(3 * a, 3 * (a + 1)) - v[i].colRange(3 * b, 3 * (b + 1));

            b++;
            if (b > 3) {
                a++;
                b = a + 1;
            }
        }
    }

    // Fill the L_6x10 matrix
    for (int i = 0; i < 6; i++) {
        l_6x10.at<double>(i, 0) = dv[0][i].dot(dv[0][i]); // ||dv[0][i]||^2
        l_6x10.at<double>(i, 1) = 2.0 * dv[0][i].dot(dv[1][i]); // 2 * dot(dv[0][i], dv[1][i])
        l_6x10.at<double>(i, 2) = dv[1][i].dot(dv[1][i]); // ||dv[1][i]||^2
        l_6x10.at<double>(i, 3) = 2.0 * dv[0][i].dot(dv[2][i]); // 2 * dot(dv[0][i], dv[2][i])
        l_6x10.at<double>(i, 4) = 2.0 * dv[1][i].dot(dv[2][i]); // 2 * dot(dv[1][i], dv[2][i])
        l_6x10.at<double>(i, 5) = dv[2][i].dot(dv[2][i]); // ||dv[2][i]||^2
        l_6x10.at<double>(i, 6) = 2.0 * dv[0][i].dot(dv[3][i]); // 2 * dot(dv[0][i], dv[3][i])
        l_6x10.at<double>(i, 7) = 2.0 * dv[1][i].dot(dv[3][i]); // 2 * dot(dv[1][i], dv[3][i])
        l_6x10.at<double>(i, 8) = 2.0 * dv[2][i].dot(dv[3][i]); // 2 * dot(dv[2][i], dv[3][i])
        l_6x10.at<double>(i, 9) = dv[3][i].dot(dv[3][i]); // ||dv[3][i]||^2
    }
}

void PnPsolver::compute_rho(double * rho)
{
  rho[0] = dist2(cws[0], cws[1]);
  rho[1] = dist2(cws[0], cws[2]);
  rho[2] = dist2(cws[0], cws[3]);
  rho[3] = dist2(cws[1], cws[2]);
  rho[4] = dist2(cws[1], cws[3]);
  rho[5] = dist2(cws[2], cws[3]);
}

void PnPsolver::compute_A_and_b_gauss_newton(const cv::Mat& L_6x10, const cv::Mat& Rho,
                                             double betas[4], cv::Mat& A, cv::Mat& b)
{
    // Ensure input dimensions
    CV_Assert(L_6x10.rows == 6 && L_6x10.cols == 10 && L_6x10.type() == CV_64F);
    CV_Assert(Rho.rows == 6 && Rho.cols == 1 && Rho.type() == CV_64F);

    // Resize A and b to proper dimensions if not already
    A = cv::Mat::zeros(6, 4, CV_64F);
    b = cv::Mat::zeros(6, 1, CV_64F);

    for (int i = 0; i < 6; i++) {
        // Access row i of L_6x10
        const double* rowL = L_6x10.ptr<double>(i);

        // Populate the A matrix
        A.at<double>(i, 0) = 2 * rowL[0] * betas[0] + rowL[1] * betas[1] + rowL[3] * betas[2] + rowL[6] * betas[3];
        A.at<double>(i, 1) = rowL[1] * betas[0] + 2 * rowL[2] * betas[1] + rowL[4] * betas[2] + rowL[7] * betas[3];
        A.at<double>(i, 2) = rowL[3] * betas[0] + rowL[4] * betas[1] + 2 * rowL[5] * betas[2] + rowL[8] * betas[3];
        A.at<double>(i, 3) = rowL[6] * betas[0] + rowL[7] * betas[1] + rowL[8] * betas[2] + 2 * rowL[9] * betas[3];

        // Populate the b vector
        b.at<double>(i, 0) = Rho.at<double>(i, 0) -
            (
                rowL[0] * betas[0] * betas[0] +
                rowL[1] * betas[0] * betas[1] +
                rowL[2] * betas[1] * betas[1] +
                rowL[3] * betas[0] * betas[2] +
                rowL[4] * betas[1] * betas[2] +
                rowL[5] * betas[2] * betas[2] +
                rowL[6] * betas[0] * betas[3] +
                rowL[7] * betas[1] * betas[3] +
                rowL[8] * betas[2] * betas[3] +
                rowL[9] * betas[3] * betas[3]
            );
    }
}


void PnPsolver::gauss_newton(const cv::Mat& L_6x10, const cv::Mat& Rho, double betas[4])
{
    const int iterations_number = 5;

    // Temporary variables for the Gauss-Newton iteration
    cv::Mat A = cv::Mat::zeros(6, 4, CV_64F); // 6x4 matrix
    cv::Mat B = cv::Mat::zeros(6, 1, CV_64F); // 6x1 matrix
    cv::Mat X = cv::Mat::zeros(4, 1, CV_64F); // 4x1 matrix (solution)

    for (int k = 0; k < iterations_number; k++) {
        // Compute A and B for the current iteration
        compute_A_and_b_gauss_newton(L_6x10, Rho, betas, A, B);

        // Solve the linear system A * X = B using QR decomposition
        cv::solve(A, B, X, cv::DECOMP_QR);

        // Update betas using the solution X
        for (int i = 0; i < 4; i++) {
            betas[i] += X.at<double>(i, 0);
        }
    }
}


void PnPsolver::qr_solve(const cv::Mat& A, const cv::Mat& b, cv::Mat& X)
{
    // Ensure the input matrix dimensions are valid
    CV_Assert(A.rows == b.rows && A.cols <= A.rows && A.type() == CV_64F && b.type() == CV_64F);

    const int nr = A.rows; // Number of rows
    const int nc = A.cols; // Number of columns

    // Clone A to work with it directly for in-place transformations
    cv::Mat R = A.clone();
    cv::Mat Q = cv::Mat::eye(nr, nr, CV_64F); // Initialize Q as identity matrix

    // Householder QR decomposition
    for (int k = 0; k < nc; k++) {
        // Extract the k-th column vector starting from row k
        cv::Mat x = R.col(k).rowRange(k, nr);

        // Compute the norm and handle numerical stability
        double norm_x = cv::norm(x, cv::NORM_L2);
        if (norm_x == 0) {
            throw std::runtime_error("Matrix A is singular, QR decomposition failed.");
        }

        // Compute Householder vector
        double alpha = (x.at<double>(0, 0) >= 0) ? -norm_x : norm_x;
        x.at<double>(0, 0) -= alpha;

        // Normalize the vector
        x /= cv::norm(x, cv::NORM_L2);

        // Update R and Q matrices
        cv::Mat H = cv::Mat::eye(nr, nr, CV_64F);
        cv::Mat Hk = H.colRange(k, nr).rowRange(k, nr);
        Hk -= 2.0 * (x * x.t());

        R = H * R;
        Q = Q * H;
    }

    // Ensure R is upper triangular
    R = R.rowRange(0, nc).colRange(0, nc);

    // Compute Qt * b
    cv::Mat Qt_b = Q.t() * b;

    // Perform explicit back substitution to solve R * X = Qt * b
    X = cv::Mat::zeros(nc, 1, CV_64F);
    for (int i = nc - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < nc; j++) {
            sum += R.at<double>(i, j) * X.at<double>(j, 0);
        }
        X.at<double>(i, 0) = (Qt_b.at<double>(i, 0) - sum) / R.at<double>(i, i);
    }
}

void PnPsolver::relative_error(double & rot_err, double & transl_err,
			  const double Rtrue[3][3], const double ttrue[3],
			  const double Rest[3][3],  const double test[3])
{
  double qtrue[4], qest[4];

  mat_to_quat(Rtrue, qtrue);
  mat_to_quat(Rest, qest);

  double rot_err1 = sqrt((qtrue[0] - qest[0]) * (qtrue[0] - qest[0]) +
			 (qtrue[1] - qest[1]) * (qtrue[1] - qest[1]) +
			 (qtrue[2] - qest[2]) * (qtrue[2] - qest[2]) +
			 (qtrue[3] - qest[3]) * (qtrue[3] - qest[3]) ) /
    sqrt(qtrue[0] * qtrue[0] + qtrue[1] * qtrue[1] + qtrue[2] * qtrue[2] + qtrue[3] * qtrue[3]);

  double rot_err2 = sqrt((qtrue[0] + qest[0]) * (qtrue[0] + qest[0]) +
			 (qtrue[1] + qest[1]) * (qtrue[1] + qest[1]) +
			 (qtrue[2] + qest[2]) * (qtrue[2] + qest[2]) +
			 (qtrue[3] + qest[3]) * (qtrue[3] + qest[3]) ) /
    sqrt(qtrue[0] * qtrue[0] + qtrue[1] * qtrue[1] + qtrue[2] * qtrue[2] + qtrue[3] * qtrue[3]);

  rot_err = min(rot_err1, rot_err2);

  transl_err =
    sqrt((ttrue[0] - test[0]) * (ttrue[0] - test[0]) +
	 (ttrue[1] - test[1]) * (ttrue[1] - test[1]) +
	 (ttrue[2] - test[2]) * (ttrue[2] - test[2])) /
    sqrt(ttrue[0] * ttrue[0] + ttrue[1] * ttrue[1] + ttrue[2] * ttrue[2]);
}

void PnPsolver::mat_to_quat(const double R[3][3], double q[4])
{
  double tr = R[0][0] + R[1][1] + R[2][2];
  double n4;

  if (tr > 0.0f) {
    q[0] = R[1][2] - R[2][1];
    q[1] = R[2][0] - R[0][2];
    q[2] = R[0][1] - R[1][0];
    q[3] = tr + 1.0f;
    n4 = q[3];
  } else if ( (R[0][0] > R[1][1]) && (R[0][0] > R[2][2]) ) {
    q[0] = 1.0f + R[0][0] - R[1][1] - R[2][2];
    q[1] = R[1][0] + R[0][1];
    q[2] = R[2][0] + R[0][2];
    q[3] = R[1][2] - R[2][1];
    n4 = q[0];
  } else if (R[1][1] > R[2][2]) {
    q[0] = R[1][0] + R[0][1];
    q[1] = 1.0f + R[1][1] - R[0][0] - R[2][2];
    q[2] = R[2][1] + R[1][2];
    q[3] = R[2][0] - R[0][2];
    n4 = q[1];
  } else {
    q[0] = R[2][0] + R[0][2];
    q[1] = R[2][1] + R[1][2];
    q[2] = 1.0f + R[2][2] - R[0][0] - R[1][1];
    q[3] = R[0][1] - R[1][0];
    n4 = q[2];
  }
  double scale = 0.5f / double(sqrt(n4));

  q[0] *= scale;
  q[1] *= scale;
  q[2] *= scale;
  q[3] *= scale;
}

} //namespace ORB_SLAM
