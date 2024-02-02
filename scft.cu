// scft.cu
//--------------------------------------------------------------
// Solves SCFT for a blend of diblock copolymer melts and homopolymers  using the 2nd-order
// pseudo-spectral method
//--------------------------------------------------------------
// Global variables:
//
// m[i] = number of mesh points in the i'th dimension
// M=m[0]*m[1]*m[2] = total number of mesh points
// Mk=m[0]*m[1]*(m[2]/2+1) = total mesh points in k-space
// NA, NB = number of steps along A and B blocks, respectively
// alpha is the ratio of homopolymer (solvent) to copolymer (lipid) length
// dsA, dsB = step size along A and B blocks, respectively
//--------------------------------------------------------------
#include<math.h>
#include<stdlib.h>
#include<complex>
#include<assert.h>
#include"mpi.h"
#include<signal.h>

#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <iostream>
#include <vector>
#include <algorithm>

#define DIM 2 // max number of histories in Anderson = DIM-1
//code doesn't currently use anderson mixing but I left this in because it is tedious to remove. also it can be useful for some modifications

using namespace std;

//const int strn=48; // # of steps on string. //should probably be in input file but it doesn't change that often
//#define strn 48
constexpr int strn = 48;

int m[3], M, Mk, NA, NB;
double dsA, dsB, pi=4*atan(1.0), phidb, D[3], *F,alpha=0.1,phis[6];
double ***DEV, ***DDEV, ***WIN;
double *expKA, *expKB, *expKA2, *expKB2;
int fix[2];

double ***q1, ***q2, **W, *cubes; //propagators, fields and stuff for cubic spline
double **phiA, **phiB, **phih, **phicB, **Wp, **Wm, **phip, **phim; //condentrations
int springsteps=0;
FILE *in, *out;
double **dWda;
char tms[50];
time_t time0, time1;
int sigg=0; //signal to kill code gently

//arrays used for cubic interpolation in string method
#define DEV_CUBES(r,j,ii) dev_cubes[ii + j*_strn + _strn*4*r]
#define CUBES(r,j,ii) cubes[ii + j*strn + strn*4*r]
double* cube_h;
double* cube_A;
double* cube_l;
double* cube_u;
double* cube_z;
double* cube_c;
double* cube_b;
double* cube_d;
double* cube_a;

//subroutines for the propagators etc. uses cuda. includes other .h files for cuda stuff
#include "density.h"

//structure to read in inputs
std::unordered_map<std::string, std::vector<double>> readParameters0(const std::string& filename) {
    std::unordered_map<std::string, std::vector<double>> params;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string key;
        if (std::getline(iss, key, '=')) {
            // Remove potential whitespace from the key
            key.erase(std::remove_if(key.begin(), key.end(), ::isspace), key.end());
            std::vector<double> values;
            double value;
            while (iss >> value) {
                values.push_back(value);
            }
            params[key] = values;
        }
    }
    
    
    return params;
}

std::unordered_map<std::string, std::vector<double>> readParameters(const std::string& filename) {
    std::unordered_map<std::string, std::vector<double>> params;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        // Ignore comments: erase everything from '#' to the end of the line
        size_t commentPos = line.find('#');
        if (commentPos != std::string::npos) {
            line.erase(commentPos);
        }
        // Trim the line to remove leading and trailing whitespace
        line.erase(line.begin(), std::find_if(line.begin(), line.end(), [](unsigned char ch) {
            return !std::isspace(ch);
        }));
        line.erase(std::find_if(line.rbegin(), line.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), line.end());
        // Continue parsing if the line is not empty after trimming
        if (!line.empty()) {
            std::istringstream iss(line);
            std::string key;
            if (std::getline(iss, key, '=')) {
                // Remove potential whitespace from the key
                key.erase(std::remove_if(key.begin(), key.end(), ::isspace), key.end());
                std::vector<double> values;
                double value;
                while (iss >> value) {
                    values.push_back(value);
                }
                params[key] = values;
            }
        }
    }
    return params;
}



// set values of parameters
void setParameters(const std::unordered_map<std::string, std::vector<double>>& params, double& chi, double& f, double& Vc, int& N, int& flag, int m[], double D[], int fix[]) {
    if (params.find("chi") != params.end()) chi = params.at("chi").front();
    if (params.find("f") != params.end()) f = params.at("f").front();
    if (params.find("Vc") != params.end()) Vc = params.at("Vc").front();
    if (params.find("N") != params.end()) N = static_cast<int>(params.at("N").front());
    if (params.find("flag") != params.end()) flag = static_cast<int>(params.at("flag").front());
    if (params.find("m") != params.end()) {
        const auto& values = params.at("m");
        for (size_t i = 0; i < values.size() && i < 3; ++i) {
            m[i] = static_cast<int>(values[i]);
        }
    }
    if (params.find("D") != params.end()) {
        const auto& values = params.at("D");
        for (size_t i = 0; i < values.size() && i < 3; ++i) {
            D[i] = values[i];
        }
    }
    if (params.find("fix") != params.end()) {
        const auto& values = params.at("fix");
        for (size_t i = 0; i < values.size() && i < 2; ++i) {
            fix[i] = static_cast<int>(values[i]);
        }
    }
}

//==============================================================
// counts the number of lines in a file
// (useful for inputs to make sure field files have the right system size)
//--------------------------------------------------------------
//int lines(char * fname){
int lines(const std::string& fname){
    int lines=0;
    FILE *fp;
    fp = fopen(fname.c_str(),"r");
    
    if (fp == nullptr) {
        // Handle error if file could not be opened
        std::cerr << "Failed to open file: " << fname << std::endl;
        return -1; // Return an error code or handle it as appropriate
    }
    
    while(!feof(fp))
    {
        char ch = fgetc(fp);
        if(ch == '\n')
        {
            lines++;
        }
    }
    fclose(fp); // Don't forget to close the file
    return lines;
}
//==============================================================
// signal handler
// used to interact wiht the schedueler (stop the code and safely exit when time runs out)
//--------------------------------------------------------------
void sig_handler(int signo)
{ 
    if (signo == SIGUSR1){
        printf("received SIGUSR1\n");
        sigg=2;
    }
    else if (signo == SIGKILL){
        printf("received SIGKILL\n");
        sigg=3;
    }
    else if (signo == SIGSTOP){
        printf("received SIGSTOP\n");
        sigg=4;
    }
    else if(signo == SIGINT){
        printf("Stop signal received.\n");
        sigg=7;
    }
}

//==============================================================
// sets the average of an array to 0 (or whatever value)
// used to reset scft fields when useful
//--------------------------------------------------------------
void avzero(double * A, int n, double zr0=0.0)
{
    double av=0;
    for( int i=0;i<n;i++) av += A[n];
    av /= n;
    for( int i=0;i<n;i++) A[i]-=av;
    //optional zet 0 to something else
    for( int i=0;i<n;i++) A[i]+=zr0;
    
}
//==============================================================
// finds the time in a string
//--------------------------------------------------------------
void tistr()
{
    long int tisec = time(NULL)-time0;
    int secs,mins, hours, days;
    int lm=60;
    int lh=60*lm;
    int ld=24*lh;
    
    days = tisec/ld;
    hours=(tisec-(days*ld))/lh;
    mins=(tisec-(days*ld)-(hours*lh))/lm;
    secs=tisec-(days*ld)-(hours*lh)-(mins*lm);
    
    if(days>0){
        sprintf(tms,"%d, %02d:%02d:%02d",days,hours,mins,secs);
    } else if(hours>0){
        sprintf(tms,"%d:%02d:%02d",hours,mins,secs);
    } else if(mins>0){
        sprintf(tms,"%d:%02d",mins,secs);
    } else {
        sprintf(tms,"%ds",secs);
    }
    
}

//==============================================================
//calculates the Eucloidian distance between two Ws
//--------------------------------------------------------------
/**/
double EDist (double **W, int ii1, int ii2){
    double dist=0;
    for(int r=0;r<M;r++) dist+= pow(W[ii1][r]-W[ii2][r],2.0);
    return sqrt(dist);
}

//==============================================================
// Given sets of inputs, fits to a function to interpolate
// REPLACE with a cubic spline
// from https://gist.github.com/svdamani/1015c5c4b673c3297309
//--------------------------------------------------------------
/**/
double cfit (double *x, double xx, double *yyp, int r){
    /** Step -1 - find where point is */
    int nm=0;
    for(int i=0;i<strn;i++){
        if(x[i]<xx) nm=i;
        else break;
    }
    double c=CUBES(r,0,nm);
    double b=CUBES(r,1,nm);
    double d=CUBES(r,2,nm);
    double a=CUBES(r,3,nm);
    double dx = xx-x[nm];
    double dx2 = dx*dx;
    double dx3=dx2*dx;
    double mm = b + 2.0*dx*c + 3.0*dx2*d;
    double yy = a + dx*b + dx2*c + dx3*d;
    yyp[0]=yy; yyp[1]=mm;
    return yy;
}
//==============================================================
// Given sets of inputs, fits to a function to interpolate
// calls cubic spline subroutines to do fit
//--------------------------------------------------------------
/**/
double fit (double *x, double xx, double *yyp, density *D2, int r){
    int wich=-1; //use fit
    if(r<0) wich=1; //do fit
    if(wich==1){ //sets up cubic spline fit
        D2-> cubefit2(x, W);
        return 0;
    }
    return cfit(x,xx,yyp,r); //uses setup to fit
}

//==============================================================
// step function
//--------------------------------------------------------------
/**/
double stepp(double x, double sig){
    if(fabs(x)<sig) return 1;
    return 0;
}
//==============================================================
// Checks if file exists
//--------------------------------------------------------------
/**/
bool fexist (const std::string& filename){
    if (FILE * file = fopen(filename.c_str(), "r"))
    {
        fclose(file);
        return true;
    }
    return false;
}
//==============================================================
// Allocates 2D array
//--------------------------------------------------------------
void malloc2d (double ***ARAY, int l1, int l2){
    *ARAY = (double **)malloc(l1 * sizeof(double *));
    for(int i=0;i<l1;i++) (*ARAY)[i] = (double *)malloc(l2 * sizeof(double));
}
void free2d(double ***array, int l1) {
    for(int i = 0; i < l1; i++) {
        free((*array)[i]); // Free each row
    }
    free(*array); // Free the array of pointers
    *array = NULL; // Set the pointer to NULL to avoid dangling pointers
}
//==============================================================
// Allocates 3D array
//--------------------------------------------------------------
void malloc3d (double ****ARAY, int l1, int l2, int l3){
    *ARAY = (double ***)malloc(l1 * sizeof(double **));
    for(int i=0;i<l1;i++) (*ARAY)[i] = (double **)malloc(l2 * sizeof(double *));
    for(int i=0;i<l1;i++) for(int j=0;j<l2;j++) (*ARAY)[i][j] = (double *)malloc(l3 * sizeof(double));
}
void new3d(double ****array, int l1, int l2, int l3) {
    *array = new double**[l1];
    for(int i = 0; i < l1; i++) {
        (*array)[i] = new double*[l2];
        for(int j = 0; j < l2; j++) {
            (*array)[i][j] = new double[l3];
        }
    }
}
void delete3d(double ****array, int l1, int l2) {
    for(int i = 0; i < l1; i++) {
        for(int j = 0; j < l2; j++) {
            delete[] (*array)[i][j];
        }
        delete[] (*array)[i];
    }
    delete[] *array;
}
//==============================================================
// 3d stuff - should wrok the same in cyl coords- note that m[2]=1;
//--------------------------------------------------------------
int xyz2r (int x, int y, int z){
    return z + y*m[2] + x*m[2]*m[1];
}
int r2x (int r){ //x or z (direction normal to circle plane)
    return r/(m[2]*m[1]);
}
int r2y (int r){  //y or r - radial direction
    return (r - r2x(r)*m[2]*m[1])/m[2] ;
}
int r2z (int r){ // z or theta
    return r%m[2];
}
double j2y (int j){
    return j*D[1]/m[1];
}
double r2yd (int r){
    int yi = r2y(r);
    return double(yi)*D[1]/m[1];
}

//==============================================================
// output to vtk
// used to visualize data
//--------------------------------------------------------------
void tovtk (const std::string& fname, int *m, double *D, double *data)
{
    int x,y,z,r;
    FILE *out;
    out = fopen(fname.c_str(),"w");
    fprintf(out,"# vtk DataFile Version 2.0\n");
    fprintf(out,"CT Density\n");
    fprintf(out,"ASCII\n\n");
    fprintf(out,"DATASET STRUCTURED_POINTS\n");
    fprintf(out,"DIMENSIONS %d %d %d\n",m[2],m[1],m[0]);
    fprintf(out,"ORIGIN 0.000000 0.000000 0.000000\n");
    fprintf(out,"SPACING %lf %lf %lf\n\n",D[2]/m[2],D[1]/m[1],D[0]/m[0]);
    fprintf(out,"POINT_DATA %ld\n",(long) m[0]*m[1]*m[2]);
    fprintf(out,"SCALARS scalars float\n");
    fprintf(out,"LOOKUP_TABLE default\n\n");
    
    for (x=0; x<m[0]; x++)
        for (y=0; y<m[1]; y++){
            for (z=0; z<m[2]; z++) {
                r = z+ y*m[2] + x*m[2]*m[1];
                fprintf(out,"%.4lf\t",data[r]);
            }
            fprintf(out,"\n");
        }
    
    fclose(out);
}
//==============================================================
// Solves field equations W_+
//--------------------------------------------------------------
int solve_field0 (double **W, const double chi, const double f, density *D2, int ii,
                  const int maxIter=1E3, const double errTol=1e-3)
{
    double lambda=0.05;
    double err=1.0, lnQ, S1, S2;
    int k, r;
    int skip=100;
    double errt=1,dlam1=1.09,dlam2=1.10;
    
    for (k=1; k<maxIter && err>errTol; k++) {
        D2->props(W[ii], &lnQ, ii, NA+NB, alpha, phidb, phis, f);
        for (r=0; r<M; r++) {
            DEV[ii][0][r]   = 0;  //remove and change loops below
            DEV[ii][0][r+M] = chi*(phiB[ii][r]+phiA[ii][r] - 1.0);
        }
        for (r=0, S1=0.0,S2=0.0; r<2*M; r++) {
            S1 += DEV[ii][0][r]*DEV[ii][0][r];
            S2 += W[ii][r]*W[ii][r];
        }
        err = pow(S1/(M),0.5);
        if(k%skip==0 || !(err==err)){
            tistr();
            printf("error0(%d) %5d/%d %.4lE/%.1lE %.2lg (Time: %s)\n",ii,k,maxIter-1,err,errTol,lambda,tms);
            assert(err==err);
        }
        
        // simple mixing
        for (r=0; r<2*M; r++) W[ii][r] = W[ii][r]+lambda*DEV[ii][0][r];
        
        if(err<errt){ //update mixing parameter during simple mixing
            lambda=min(0.1,lambda*dlam1);
        } else {
            lambda=max(0.005,lambda/dlam2);
        }
        errt=err;
    }
    
    return k;
}

//==============================================================
// Solves field equations for W_+ and W_-
//--------------------------------------------------------------
int solve_field (double **W, const double chi, const double f, density *D2, double *alf, int *doiis, int *ndo, int *whois, int procid, int numprocs, int doii=-1,
                 const int maxIter=1E1, const double errTol=1e-3)
{
    
    double lambda=0.05;
    double err=1, lnQ, S1, S2,errs[strn], errs3[strn],err3,stepav;
    double normDER[strn], normDEV[strn], fitup, fitups[strn];
    int    k, r;
    int skip=50;
    double errt=1,dlam1=1.015,dlam2=1.016;
    double y01 = m[1]/2 + 17;
    double y02 = m[1]/4 - 0;
    double eps=0.5;
    std::string outfl;
    double Del[strn-1],Delsum;
    double x[strn],yyp[2],dotWs[strn], dotW,xref[strn],dotWs2[strn];
    int ii, steps[strn];
    
    for(ii=0;ii<strn;ii++) steps[ii]=0;
    
    //'spring' steps sets a parabolic potential between replicas, equivalent to having a 'spring' between them - useful for initializing configurations from rough initial guesses or from a coarse string. Often useful at the beginning of a run but produces incorrect results if used through the whole simulation. better to run for a bit, then turn off and do redist + perp
    //'redist' redistributes points using a cubic spline
    //'perp' updartes the string only perpendicular to its trajectory.
    int dospring=1,doredist=0,doperp=0;
    if(doii>-1) dospring=0,doredist=0,doperp=0; //do not do these if doing just one replica
    
    //useful to turn on/off in order to update or not update first/last point
    // this is useful when studying the trajectory between unstable configurations
    if(fix[0]==1)doiis[0]=0;
    if(fix[1]==1)doiis[strn-1]=0;
    
    printf("Solving fields with %d springsteps. (procid=%d)\n",springsteps,procid);
    
    for (k=0; k<maxIter && err>errTol; k++) {
        if(k>=springsteps && doii==-1) {dospring=0; doredist=1; doperp=1;}
        
        for(ii=0;ii<strn;ii++) for (r=0; r<2*M; r++) DEV[ii][0][r]=0;
        for(ii=0;ii<strn;ii++){
            steps[ii]=0;
            if((doiis[ii]==1 && doii==-1) || doii==ii){ //MPI
                steps[ii]++;
                //solve W_+ (at least partially) for each step in W_-
                //if you don't do this, it can become unstable and/or otherwise mess up
                steps[ii]+=solve_field0(W, chi, f, D2, ii, 20, errTol);
                D2->props(W[ii], &lnQ, ii, NA+NB, alpha, phidb, phis, f);
                for (r=0; r<M; r++) {
                    DEV[ii][0][r] = (chi*(phiB[ii][r]-phiA[ii][r]) - 2.0*W[ii][r]);
                    if(ii>0 && ii<strn-1 && dospring==1) DEV[ii][0][r] += eps*(W[ii+1][r] + W[ii-1][r] - 2.0*W[ii][r]);
                    DEV[ii][0][r+M] = chi*(phiB[ii][r]+phiA[ii][r] - 1.0); //note that this is chi * the usual error/change in W_+ that people usually use (and the error is usually given as the deviation of the total density, not chi*that). I set it like this so it looks the same as the usual W_A and W_B updates. It is arbitrary, but this way overestimates the W_+ error (relative to the usual W_+ approach).
                }
            }
        }
        
        if(doii==-1){
            Delsum=0;
            for(ii=0;ii<strn-1;ii++){ //distances
                Del[ii]=0;
                for(r=0; r<M; r++) Del[ii]+=pow(W[ii][r]-W[ii+1][r],2.0);
                Del[ii]=sqrt(Del[ii]);
                Delsum+=Del[ii];
            }
            for(ii=0;ii<strn-1;ii++) Del[ii]/=Delsum; //sum distances
            x[0]=0; x[strn-1]=1; //ends
            for(ii=1;ii<strn-1;ii++) x[ii]=x[ii-1]+Del[ii-1]; //actual positions
            for(ii=0;ii<strn;ii++) xref[ii] = (1.0*ii)/(strn-1.0); //reference positions
        }
        
        if(doii==-1){
            for(ii=0;ii<strn;ii++){
                for (r=0, S1=0.0, S2=0.0; r<2*M; r++) {
                    S1 += DEV[ii][0][r]*DEV[ii][0][r];
                    S2 += W[ii][r]*W[ii][r];
                }
                errs[ii] = pow(S1/(2*M),0.5);
            }
        } else {
            for(ii=0;ii<strn;ii++) errs[ii]=0;
            for (r=0, S1=0.0, S2=0.0; r<2*M; r++) {
                S1 += DEV[doii][0][r]*DEV[doii][0][r];
                S2 += W[doii][r]*W[doii][r];
            }
            errs[doii] = pow(S1/(2*M),0.5);
        }
        
        if(doperp==1){
            fit(x, x[0], yyp,D2,-1);
            for(r=0; r<M; r++) {
                for(ii=0;ii<strn;ii++){
                    fit(x, x[ii], yyp,D2,r);
                    dWda[ii][r]=yyp[1];
                }
            }
            //find normalizations of derivatives = normDER[ii]
            for(ii=0;ii<strn;ii++){
                normDER[ii]=0;
                for(r=0; r<M; r++) normDER[ii]+=dWda[ii][r]*dWda[ii][r];
                normDER[ii]=sqrt(normDER[ii]);
            }
            //find norm of DEV[ii][0][r]  = normDEV[ii]
            for(ii=0;ii<strn;ii++){
                normDEV[ii]=0;
                if(doiis[ii]==1)
                    for(r=0; r<M; r++) normDEV[ii]+=DEV[ii][0][r]*DEV[ii][0][r];
                normDEV[ii]=sqrt(normDEV[ii]);
            }
            //find dotW[ii] = (DEV[ii][0][r]/normDEV) dot (detivative/normDER)
            for(ii=0;ii<strn;ii++){
                dotWs[ii]=0;
                if(doiis[ii]==1)
                    for(r=0; r<M; r++) dotWs[ii] += (DEV[ii][0][r]/normDEV[ii]) * (dWda[ii][r]/normDER[ii]);
            }
            //subtract parallel part (dotW[ii]*detivative[ii][r]*normDEV[ii]) from DEV[ii][r]
            for(ii=0;ii<strn;ii++) errs3[ii]=0;
            for(ii=1;ii<strn-1;ii++){ //onlymiddle steps
                if(doiis[ii]==1 && normDER[ii]>0 && normDEV[ii]>0){
                    for(r=0; r<M; r++) DEV[ii][0][r] -= (dotWs[ii]/normDER[ii])*dWda[ii][r]*normDEV[ii];
                    for(r=0; r<2*M; r++) errs3[ii]+= DEV[ii][0][r]*DEV[ii][0][r];
                    errs3[ii] = pow(errs3[ii]/(2*M),0.5);
                } else if(doiis[ii]==1){
                    printf("fuck (%d)... %d %d %lf %lf\n",procid,k,ii,normDER[ii],normDEV[ii]);
                }
            }
            for(ii=0;ii<strn;ii++){
                dotWs2[ii]=0;
                if(doiis[ii]==1)
                    for(r=0; r<M; r++) dotWs2[ii] += (DEV[ii][0][r]/normDEV[ii]) * (dWda[ii][r]/normDER[ii]);
            }
        } else for(ii=0; ii<strn; ii++) errs3[ii]=errs[ii];
        
        
        errs3[0]=errs[0]; errs3[strn-1]=errs[strn-1]; //set ends to err (only middle points set above)
        
        
        // simple mixing with fixed mixing parameter
        for(ii=0;ii<strn;ii++){
            for (r=0; r<2*M; r++) W[ii][r] = W[ii][r]+lambda*DEV[ii][0][r];
        } //ii
        
        //MPI
        //exchange Ws here
        if(numprocs>1){
            for(ii=0;ii<strn;ii++)
                MPI_Bcast(&errs[ii], 1, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);
            
            for(ii=0;ii<strn;ii++) MPI_Bcast(&steps[ii], 1, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);
            for(ii=0;ii<strn;ii++) MPI_Bcast(&errs3[ii], 1, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);
            
            for(ii=0;ii<strn;ii++)
                MPI_Bcast(W[ii], M, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);
            
            for(ii=0;ii<strn;ii++)
                MPI_Bcast(&dotWs[ii], 1, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);
            
            for(ii=0;ii<strn;ii++)
                MPI_Bcast(&dotWs2[ii], 1, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);
            
            for(ii=0;ii<strn;ii++)
                MPI_Bcast(&normDEV[ii], 1, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
        }
        stepav=0;
        for(ii=0;ii<strn;ii++) stepav+=steps[ii];
        stepav /= strn;
        
        dotW=0; for(ii=1;ii<strn-1;ii++) dotW+=dotWs[ii]; //total dotW
        
        
        //find distance along string... again
        if(doii==-1){
            Delsum=0;
            for(ii=0;ii<strn-1;ii++){
                Del[ii]=0;
                for(r=0; r<M; r++) Del[ii]+=pow(W[ii][r]-W[ii+1][r],2.0);
                Del[ii]=sqrt(Del[ii]);
                Delsum+=Del[ii];
            }
            
            //move Ws to fit
            fitup=0;
            for(ii=0;ii<strn-1;ii++) Del[ii]/=Delsum;
            x[0]=0; x[strn-1]=1;
            for(ii=1;ii<strn-1;ii++) x[ii]=x[ii-1]+Del[ii-1];
            for(ii=0;ii<strn;ii++) alf[ii]=x[ii];
            for(ii=0;ii<strn;ii++) fitups[ii] = x[ii]-xref[ii];
            for(ii=1;ii<strn-1;ii++) fitup += pow(fitups[ii],2.0);
            fitup /= strn-2.0;
            fitup = sqrt(fitup);
            fitup /= strn-1.0; //scale by spacing between
        }
        
        if(doredist==1){
            fit(x, x[0], yyp,D2,-1); //setup coefficients for spline
            for (r=0; r<M; r++){ //points in space (each point is fit to a curve along the ii direction)
                for(ii=1;ii<strn-1;ii++) //ii is the aaxis along which teh fit is done
                    W[ii][r] += (fit(x, xref[ii], yyp,D2,r) - W[ii][r])*lambda; //'fit' gets the value at xref[ii] but the code slowly increments (lambda part) otherwise it can become unstable in some circumstances
            } //r
        }
        //updated
        
        if(procid==0) //periodic output from proc 0
            if(k%skip==0 || k==maxIter-1 || k==-1 || !(err>errTol) || sigg==7) {
                tistr();
                if(doii==-1){
                    double errtot=0; for(ii=0;ii<strn;ii++) errtot+=errs3[ii]*errs3[ii]/strn;
                    errtot=sqrt(errtot);
                    printf("error  %5d/%d %.4lE/%.1lE %.2lg.\tEs: %.2lE %.2lE ... %.2lE ... %.2lE %.2lE... %.2lg.. %.1lf (Time: %s)\n",k,maxIter,errtot,errTol,lambda, errs3[0],errs3[1],errs3[strn/2],errs3[strn-2],errs3[strn-1],fitup,stepav,tms);
                } else {
                    printf("error[%d]  %5d/%d %.4lE/%.1lE %.2lg.\tE: %.2lE ... %d (Time: %s)\n",doii,k,maxIter,err3,errTol,lambda, errs[doii],steps[ii],tms);
                }
                assert(err==err); //kill code if error is nan
                
                out=fopen("check","w"); //output errors to a file to check stuff
                fprintf(out,"%d %lf %lf\n",k,err3,lambda);
                for(ii=0;ii<strn;ii++) fprintf(out,"%d %.12lf %.12lf\n",ii,errs[ii],errs3[ii]);
                fprintf(out,"\n");
                fclose(out);
            }
        //regularly output fields and A monomer concentration
        if(k%(skip)==0 || k==maxIter-1 || k==-1 || !(err>errTol) || sigg==7){
            for(ii=0;ii<strn;ii++){
                if(((doiis[ii]==1 && doii==-1) || doii==ii) && err==err){
                    outfl = "rhoA_" + std::to_string(ii) + ".vtk";
                    tovtk(outfl, m, D, phiA[ii]);
                    
                    outfl = "win" + std::to_string(ii);
                    out=fopen(outfl.c_str(),"w");
                    for (r=0;r<M;r++) fprintf(out,"%.6lf %.6lf\n",W[ii][r],W[ii][r+M]);
                    fclose(out);
                }
            }
        }
        
        if(err<errt){ //update mixing parameter during simple mixing
            lambda=min(0.2,lambda*dlam1);
        } else {
            lambda=max(0.05,lambda/dlam2);
        }
        errt=err;
        
        if(sigg==7) {
            MPI_Barrier(MPI_COMM_WORLD);
            printf("Time's up (procid=%d).\n",procid);
            break;
        }
        
    }
    
    return k;
}




/**/
//==============================================================
// Key variables in main:
//
// W = array containing wA, wB, phiA, phiB
// r = (x*m[1]+y)*m[2]+z = array position for (x,y,z)
// FE = free energy
// lnQ = logQ terms in free energy (calculated by 'props')
// chi = chi*N
// f = volume fraction of the A block
// N = number of contour steps in lipid
//
//--------------------------------------------------------------
int main (int argc, char *argv[])
{
    double chi, f, lnQ;
    int    x, y, z, r, N, flag,ii=0;
    time0 = time(NULL); //code start time
    
    int seed = (12345+time(NULL));
    srand(seed);
    
    //test gpu
    printf("Hello World from host!\n");
    print_from_gpu<<<1,1>>>();
    cudaDeviceSynchronize();
    
    if (signal(SIGINT, sig_handler) == SIG_ERR)
        printf("\ncan't catch SIGINT\n");
    
    
    //initialize MPI
    int procid=0, numprocs=1,ierr=0;
    ierr = MPI_Init (NULL,NULL);
    if ( ierr != 0 )
    {
        printf ( "\n" );
        printf ( "HELLO_MPI - Fatal error!\n" );
        printf ( "  MPI_Init returned nonzero IERR.\n" );
        exit ( 1 );
    }
    //initialize mpi
    ierr = MPI_Comm_size ( MPI_COMM_WORLD, &numprocs );
    ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &procid );
    printf("Hello from procid %d/%d\n",procid,numprocs);
    
    
    
    std::string outfl;
    std::string winf;
    double Vc;
    
    std::string finc;
    
    if (argc > 1) {
        finc = argv[1];
    } else {
        finc = "input.dat";
    }
    printf("Reading input from %s\n",finc.c_str());
    
    auto params = readParameters(finc); //read in parameters
    setParameters(params, chi, f, Vc, N, flag, m, D, fix); //set parameter values

    double V = D[2]*D[1]*D[0];
    phidb = Vc; //GC
    
    printf("Input parameters:\n");
    printf("%lf %lf %lf (%lf)\n",chi,f,Vc,phidb);
    printf("%d %d %d %d\n",N,m[0],m[1],m[2]);
    printf("%lf %lf %lf\n",D[0],D[1],D[2]);
    printf("%d %d\n",fix[0],fix[1]);
    printf("%d\n",flag);
    
    
    //Decide which processors do what
    int *doiis, *ndo, *whois, before=0, bf2=0;
    doiis = new int[strn];
    whois = new int[strn];
    ndo = new int[numprocs];
    for(ii=0;ii<strn;ii++) doiis[ii]=0;
    if(numprocs>=strn) { //if enough procs to do a run on each cpu, do that
        if(procid<strn) doiis[procid]=1; //doiis is list of string positions that this processor will do. exclude procids greater than string length.. in case there are any
        for(ii=0;ii<numprocs;ii++) {if(ii<strn) ndo[ii]=1; else ndo[ii]=0;} //array of how many string elements each will do
    } else {
        int ndo0 = strn/numprocs; //baseline number of string elements
        for(ii=0;ii<numprocs;ii++) ndo[ii]=ndo0; //set that in array of numbers to do
        for(ii=0;ii<(strn-numprocs*ndo0);ii++) ndo[ii]++; // fill in any let over
        for(ii=0;ii<procid;ii++) before += ndo[ii]; //count numbers done b previous processors
        for(ii=0;ii<ndo[procid];ii++) doiis[before+ii]=1; //do these iis
        for(ii=0;ii<numprocs;ii++) {
            for(int iii=0;iii<ndo[ii];iii++)
            {whois[bf2]=ii; bf2++;}
            
        }
    }
    
    M = m[0]*m[1]*m[2];
    Mk = m[0]*m[1]*(m[2]/2+1);
    
    NA=(int) round (N*f), NB=N-NA;
    f = double(NA)/N;
    dsA = f/NA, dsB = (1-f)/NB;
    
    density *D2 = new density(N, D);
    
    //cubic spline stuff
    cube_h = new double[strn];
    cube_A = new double[strn];
    cube_l = new double[strn];
    cube_u = new double[strn];
    cube_z = new double[strn];
    cube_c = new double[strn];
    cube_b = new double[strn];
    cube_d = new double[strn];
    cube_a = new double[strn];
    
    cubes = new double[M * strn * 4];
    
    new3d(&DEV, strn, DIM, 2 * M);
    new3d(&DDEV, strn, DIM, 2 * M);
    new3d(&WIN, strn, DIM, 2 * M);
    
    new3d(&q1, strn, N + 1, M);
    new3d(&q2, strn, N + 1, M);
    
    //Fields and concentrations and other arrays of potentially useful stuff
    malloc2d(&W,strn,15*M);
    phiA = (double **)malloc(strn * sizeof(double *));
    phiB = (double **)malloc(strn * sizeof(double *));
    phih = (double **)malloc(strn * sizeof(double *));
    phicB = (double **)malloc(strn * sizeof(double *));
    Wp = (double **)malloc(strn * sizeof(double *));
    Wm = (double **)malloc(strn * sizeof(double *));
    phip = (double **)malloc(strn * sizeof(double *));
    phim = (double **)malloc(strn * sizeof(double *));
    dWda = (double **)malloc(strn * sizeof(double *));
    //new2d(&W, strn, 15 * M);
    
    for(ii=0;ii<strn;ii++){
        phiA[ii]=W[ii]+2*M;
        phiB[ii]=W[ii]+3*M;
        phih[ii]=W[ii]+4*M;
        phicB[ii]=W[ii]+5*M;
        Wp[ii]= W[ii]+6*M;
        Wm[ii]= W[ii]+7*M;
        phip[ii]= W[ii]+8*M;
        phim[ii]= W[ii]+9*M;
        dWda[ii]=W[ii]+10*M;
    }
    
    ii=0;
    
    int cpto0=0;
    
    if (flag==1) {
        for(ii=0;ii<strn;ii++){
            winf = "win" + std::to_string(ii);
            if(fexist(winf) && lines(winf)==m[0]*m[1]*m[2]){
                in=fopen(winf.c_str(),"r");
                for (r=0;r<M;r++) fscanf(in,"%lf %lf",&W[ii][r],&W[ii][r+M]);
                fclose(in);
                if(doiis[ii]==1) printf("win%d read.\n",ii);
            } else if(!(lines(winf)==m[0]*m[1]*m[2])) { //deal with file errors, in case files have the wrong number of lines
                printf("win%d is too small: %d.\n",ii,lines(winf));
                springsteps +=1;
                if(ii!=0){
                    for (r=0;r<3*M;r++) W[ii][r] = W[ii-1][r];
                } else {
                    cpto0=1; //simple modification to deal with errors reading in files
                }
                
            }
        }
        if(cpto0==1){
            for (r=0;r<2*M;r++) W[0][r] = W[1][r];
        }
    } else {
        for (x=0; x<m[0]; x++)
            for (y=0; y<m[1]; y++)
                for (z=0; z<m[2]; z++) { //initial W here - right now parallel to walls
                    r = (x*m[1]+y)*m[2]+z;
                    for(ii=0;ii<strn;ii++){
                        W[ii][r] = 0;
                        W[ii][r+M] = 0;
                    }
                    double rr2 = (z-m[2]/2.0)*(z-m[2]/2.0) + (y-m[1]/2.0)*(y-m[1]/2.0);
                    double rr = sqrt(rr2);
                    ii=0;
                    //outer I //just a potential initialization
                    if(x<2*m[0]/5 || x>3*m[0]/5 || 1==1)
                        W[ii][r] += -0.5*chi*exp(-1.0*(pow((rr-m[1]/4.0)/8.0,2.0)));
                    //inner I
                    if(x<m[0]/3-8 || x>2*m[0]/3+8 || 1==1)
                        W[ii][r] += -0.5*chi*exp(-1.0*(pow((rr-m[1]/8.0)/8.0,2.0)));
                }
    }
    
    printf("Setup\n");
    
    int nsterp=4000; //max number of steps in convergence
    double alf[strn];
    
    for(ii=0;ii<strn;ii++) if(doiis[ii]==1) solve_field0(W, chi, f, D2, ii, 5E3, 1E-4); //solve W_+ before we start
    for(ii=0;ii<strn;ii++){ //output initial state (with converged W_+)
        if(doiis[ii]==1){
            outfl = "rhoA_" + std::to_string(ii) + ".vtk";
            tovtk(outfl, m, D, phiA[ii]);
            outfl = "win" + std::to_string(ii);
            out=fopen(outfl.c_str(),"w");
            for (r=0;r<M;r++) fprintf(out,"%.6lf %.6lf\n",W[ii][r],W[ii][r+M]);
            fclose(out);
        }
    }
    solve_field(W,chi,f,D2,alf,doiis,ndo,whois,procid,numprocs,-1,nsterp); //solve W_-
    
    for(ii=0;ii<strn;ii++)
        MPI_Bcast(W[ii], M*2, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    printf("Fields solved.\n");
    
    //calculate free energies
    double FEs0[strn],FEs1[strn],FEs1a[strn],FEs[strn],wa,wb,phctot[strn], FE=0;
    for(ii=0;ii<strn;ii++){
        if(doiis[ii]==1) solve_field0(W, chi, f, D2, ii, 1E2, 1E-4); //solve W_+
        D2->props(W[ii], &lnQ, ii, NA+NB, alpha, phidb, phis, f);
        FEs0[ii] = -lnQ;
        FEs1[ii]=0;
        FEs1a[ii]=0;
        phctot[ii]=0;
        for(int r=0;r<M;r++) {
            wa=W[ii][r]+W[ii][r+M]; wb=W[ii][r+M]-W[ii][r];//can be calculated using W_- and W_+ directly. doesn't make a difference. I forgot why I did it like this
            FEs1[ii] += (chi*phiA[ii][r]*phiB[ii][r]-wa*phiA[ii][r]-wb*phiB[ii][r])/M;
            FEs1a[ii] += (W[ii][r]*W[ii][r]/chi - W[ii][r+M])/M;
            phctot[ii]+=phiA[ii][r]/(M*f);
        }
        FEs[ii] = FEs0[ii] + FEs1[ii];
        FE+=FEs[ii]; //rudementary error check to make sure nothing is nan
    }
    for(ii=0;ii<strn;ii++){
        MPI_Bcast(&FEs[ii], 1, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);
        MPI_Bcast(&FEs0[ii], 1, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);
        MPI_Bcast(&FEs1[ii], 1, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);
        MPI_Bcast(&FEs1a[ii], 1, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    printf("FE caculated\n");
    
    
    if(FE==FE && procid==0){ //only output field if stuff isn't nan
        outfl = "FEs";
        out = fopen(outfl.c_str(),"w");
        //for(ii=0;ii<strn;ii++) fprintf(out,"%lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\n",alf[ii],FEs[ii],FEs0[ii],FEs1[ii],phctot[ii]); //if you want the breakdown of free energies.
        for(ii=0;ii<strn;ii++) fprintf(out,"%lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\n",alf[ii],FEs[ii],FEs0[ii],FEs1[ii],FEs1a[ii],phctot[ii]);
        //for(ii=0;ii<strn;ii++) fprintf(out,"%lf\t%.10lf\t%.10lf\n",alf[ii],FEs[ii],phctot[ii]);
        fclose(out);
        ii=0;
        
        //output fields
        for(ii=0;ii<strn;ii++){
            outfl = "win" + std::to_string(ii);
            out=fopen(outfl.c_str(),"w");
            for (r=0;r<M;r++) fprintf(out,"%.6lf %.6lf\n",W[ii][r],W[ii][r+M]);
            fclose(out);
        }
        //output vtks - just output rhoA and cB. rhoB is just 1-rhoA so that's not too useful.
        for(ii=0;ii<strn;ii++){
            outfl = "rhoA_" + std::to_string(ii) + ".vtk";
            tovtk(outfl, m, D, phiA[ii]);
            
            outfl = "rhocB_" + std::to_string(ii) + ".vtk";
            tovtk(outfl, m, D, phicB[ii]);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    printf("vtks and wins printed\n");
    
    
    //Deallocating arrays
    
    //Deallocating 1D arrays:
    delete[] cubes;
    delete[] cube_h;
    delete[] cube_A;
    delete[] cube_l;
    delete[] cube_u;
    delete[] cube_z;
    delete[] cube_c;
    delete[] cube_b;
    delete[] cube_d;
    delete[] cube_a;
    
    delete3d(&q1, strn, N + 1);
    delete3d(&q2, strn, N + 1);
    delete3d(&DEV, strn, DIM);
    delete3d(&DDEV, strn, DIM);
    delete3d(&WIN, strn, DIM);
    
    // Deallocating W
    free2d(&W, strn);
    //phiA etc could be set to NULL here. But I don't think it matters.
    
    
    
    tistr();
    printf("Done (%d) (Time: %s)\n",procid,tms);
}

