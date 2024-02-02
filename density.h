//density.h

#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include "GPU_error.h"
#include "GPU_kernels.h"

#define imin(a,b) (a<b?a:b)



class step
//------------------------------------------------------------
// steps propagators: in=q(r,s) and out=q(r,s+ds)
//------------------------------------------------------------
{
    // CUFFT DEVICE MEMORY VARIABLES //
    cufftDoubleReal *dev_qs1, *dev_qs0;		// real-space data
    cufftDoubleComplex *dev_h;				// k-space data
    
    // CUFFT PLANS //
    cufftHandle cufft_plan, cufft_planb;
    
    // HOST LOOKUP TABLES //
    double *_k_sq;
    
    // DEVICE LOOKUP TABLES //
    double *dev_expKsq, *dev_expKsq2, *dev_k_sq;
    double *dev_expWds_p1, *dev_expWds2_p1;
    double *dev_expWds_m1, *dev_expWds2_m1;
    
    // FOR CUFFTPLANMANY() //
    int _rank = 3;							// 3D transform
    int _batch = 2;							// q and q^dagger
    
    
public:
    //------------------------------------------------------------
    
    step(double *DA)		// CONSTRUCTOR //
    {
        // ALLOCATE DEVICE LOOKUP TABLES FOR CURRENT FIELD //
        HANDLE_ERROR(cudaMalloc((void**)&dev_expKsq ,		Mk * sizeof(double)));
        HANDLE_ERROR(cudaMalloc((void**)&dev_expKsq2,		Mk * sizeof(double)));
        HANDLE_ERROR(cudaMalloc((void**)&dev_expWds_p1,		M  * sizeof(double)));
        HANDLE_ERROR(cudaMalloc((void**)&dev_expWds2_p1,	M  * sizeof(double)));
        HANDLE_ERROR(cudaMalloc((void**)&dev_expWds_m1,		M  * sizeof(double)));
        HANDLE_ERROR(cudaMalloc((void**)&dev_expWds2_m1,	M  * sizeof(double)));
        
        // ALLOCATE LOCAL _K_SQ LOOKUP ARRAY AND POPULATE //
        _k_sq = new double[Mk];
        setup_k_sq(DA);
        
        // ALLOCATE DEVICE LOOKUP TABLE K_SQ //
        HANDLE_ERROR(cudaMalloc((void**)&dev_k_sq, Mk * sizeof(double)));
        
        // PASS THE K_SQ LOOKUP TABLE FROM HOST TO DEVICE //
        HANDLE_ERROR(cudaMemcpy(dev_k_sq, _k_sq, Mk * sizeof(double), cudaMemcpyHostToDevice));
        
        // CUFFT DEVICE MEMORY ALLOCATION //
        HANDLE_ERROR(cudaMalloc((void**)&dev_qs0, _batch * M  * sizeof(cufftDoubleReal)));
        HANDLE_ERROR(cudaMalloc((void**)&dev_qs1, _batch * M  * sizeof(cufftDoubleReal)));
        HANDLE_ERROR(cudaMalloc((void**)&dev_h  , _batch * Mk * sizeof(cufftDoubleComplex)));
        
        // CUFFTPLANMANY() //
        HANDLE_ERROR(cufftPlanMany(&cufft_plan, _rank, m,
                                   NULL, 1, 0,
                                   NULL, 1, 0,
                                   CUFFT_D2Z, _batch));
        HANDLE_ERROR(cufftPlanMany(&cufft_planb, _rank, m,
                                   NULL, 1, 0,
                                   NULL, 1, 0,
                                   CUFFT_Z2D, _batch));
    }
    
    //------------------------------------------------------------
    
    void setup_k_sq(double *DA) {		// SET UP THE K_SQ[MK] ARRAY TO SEPARATE FROM FTMC.CU //
        int k;
        
        for (int k0 = -(m[0] - 1) / 2; k0 <= m[0] / 2; k0++) {
            const int K0 = (k0<0) ? (k0 + m[0]) : k0;
            double A0 = k0*k0 / (DA[0] * DA[0]);
            for (int k1 = -(m[1] - 1) / 2; k1 <= m[1] / 2; k1++) {
                const int K1 = (k1<0) ? (k1 + m[1]) : k1;
                double A1 = A0 + k1*k1 / (DA[1] * DA[1]);
                for (int k2 = 0; k2 <= m[2] / 2; k2++) {
                    double A2 = A1 + k2*k2 / (DA[2] * DA[2]);
                    k = k2 + (m[2] / 2 + 1)*(K1 + m[1] * K0);
                    _k_sq[k] = 4 * pi*pi*A2;
                }
            }
        }
    }
    
    //------------------------------------------------------------
    
    ~step()		// DESTRUCTOR //
    {
        delete[] _k_sq;
        HANDLE_ERROR(cufftDestroy(cufft_planb));
        HANDLE_ERROR(cufftDestroy(cufft_plan));
        HANDLE_ERROR(cudaFree(dev_h));
        HANDLE_ERROR(cudaFree(dev_qs1));
        HANDLE_ERROR(cudaFree(dev_qs0));
        HANDLE_ERROR(cudaFree(dev_expKsq));
        HANDLE_ERROR(cudaFree(dev_expKsq2));
        HANDLE_ERROR(cudaFree(dev_expWds_p1));
        HANDLE_ERROR(cudaFree(dev_expWds2_p1));
        HANDLE_ERROR(cudaFree(dev_expWds_m1));
        HANDLE_ERROR(cudaFree(dev_expWds2_m1));
        HANDLE_ERROR(cudaFree(dev_k_sq));
    }
    
    //------------------------------------------------------------
    
    void prepareFactors (const double ds, const double *dev_W)
    {
        /* Laplacian part of diffusion equation */
        Prepare_dev_expKsq2_expKsq<<<(Mk+threadsPerBlock-1)/threadsPerBlock, threadsPerBlock>>>(dev_expKsq2, dev_expKsq, dev_k_sq, ds, M, Mk);
        
        // PREPARE THE FACTOR expW(r) ON THE DEVICE //
        Prepare_dev_dev_expWds2_dev_expWds << <(M + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> >(dev_expWds2_p1, dev_expWds_p1, dev_expWds2_m1, dev_expWds_m1, dev_W, ds, M);
    }
    
    //------------------------------------------------------------
    
    void do_step (const double *dev_in, double *dev_out, int s, int n1, int n2)
    // steps propagators: dev_in=q(r,s) and dev_out=q(r,s+ds) point to device memory //
    {
        
        
        // (1) FIRST HALF STEP //
        // Mult<<<(M + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock>>>(dev_qs0, dev_in, dev_expWds2, M);
        // q
        if (s < n1) {
            Mult << <(M + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> >(dev_qs0, dev_in, dev_expWds2_p1, M);
        } else {
            Mult << <(M + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> >(dev_qs0, dev_in, dev_expWds2_m1, M);
        }
        // q^dagger
        if (s < n2) {
            Mult << <(M + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> >(dev_qs0 + M, dev_in + M, dev_expWds2_m1, M);
        } else {
            Mult << <(M + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> >(dev_qs0 + M, dev_in + M, dev_expWds2_p1, M);
        }
        
        // (2) PERFORM GPU FOURIER TRANSFORM ON dev_qs0 TO GET dev_h (CUFFTPLANMANY) //
        HANDLE_ERROR(cufftExecD2Z(cufft_plan, dev_qs0, dev_h));
        
        // (4) MULTIPLY CUFFTDOUBLECOMPLEX BY A DOUBLE //
        Mult_self << <(Mk + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> >(dev_h, dev_expKsq2, Mk);
        Mult_self << <(Mk + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> >(dev_h + Mk, dev_expKsq2, Mk);
        
        // (6) NEED TO PERFORM THE GPU FOURIER TRANSFORM ON dev_h TO GET dev_qs0 //
        HANDLE_ERROR(cufftExecZ2D(cufft_planb, dev_h, dev_qs0));
        
        // SECOND HALF STEP //
        // q
        if (s < n1) {
            Mult_self << <(M + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> >(dev_qs0, dev_expWds_p1, M);
        } else {
            Mult_self << <(M + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> >(dev_qs0, dev_expWds_m1, M);
        }
        // q^dagger
        if (s < n2) {
            Mult_self << <(M + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> >(dev_qs0 + M, dev_expWds_m1, M);
        } else {
            Mult_self << <(M + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> >(dev_qs0 + M, dev_expWds_p1, M);
        }
        
        // (9) PERFORM GPU FOURIER TRANSFORM ON dev_qs0 TO GET dev_h (CUFFTPLANMANY) //
        HANDLE_ERROR(cufftExecD2Z(cufft_plan, dev_qs0, dev_h));
        
        // (11) MULTIPLY CUFFTDOUBLECOMPLEX BY A DOUBLE //
        Mult_self<<<(Mk + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock>>>(dev_h, dev_expKsq2, Mk);
        Mult_self << <(Mk + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> >(dev_h + Mk, dev_expKsq2, Mk);
        
        // (13) NEED TO PERFORM THE GPU FOURIER TRANSFORM ON dev_h TO GET dev_qs0 (CUFFTPLANMANY) //
        HANDLE_ERROR(cufftExecZ2D(cufft_planb, dev_h, dev_qs0));
        
        // FULL STEP //
        //Mult<<<(M + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock>>>(dev_qs1, dev_in, dev_expWds, M);
        // q
        if (s < n1) {
            Mult << <(M + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> >(dev_qs1, dev_in, dev_expWds_p1, M);
        } else {
            Mult << <(M + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> >(dev_qs1, dev_in, dev_expWds_m1, M);
        }
        // q^dagger
        if (s < n2) {
            Mult << <(M + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> >(dev_qs1 + M, dev_in + M, dev_expWds_m1, M);
        } else {
            Mult << <(M + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> >(dev_qs1 + M, dev_in + M, dev_expWds_p1, M);
        }
        
        // (16) PERFORM GPU FOURIER TRANSFORM ON dev_qs1 TO GET dev_h (CUFFTPLANMANY) //
        HANDLE_ERROR(cufftExecD2Z(cufft_plan, dev_qs1, dev_h));
        
        // (18) MULTIPLY CUFFTDOUBLECOMPLEX BY A DOUBLE //
        Mult_self << <(Mk + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> >(dev_h, dev_expKsq, Mk);
        Mult_self << <(Mk + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> >(dev_h + Mk, dev_expKsq, Mk);
        
        // (20) NEED TO PERFORM THE GPU FOURIER TRANSFORM ON dev_h TO GET dev_qs1 (CUFFTPLANMANY) //
        HANDLE_ERROR(cufftExecZ2D(cufft_planb, dev_h, dev_qs1));
        
        /* Richardson extrapolation */
        // q
        if (s < n1) {
            Richardson << <(M + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> >(dev_out, dev_qs0, dev_qs1, dev_expWds2_p1, dev_expWds_p1, M);
        } else {
            Richardson << <(M + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> >(dev_out, dev_qs0, dev_qs1, dev_expWds2_m1, dev_expWds_m1, M);
        }
        // q^dagger
        if (s < n2) {
            Richardson << <(M + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> >(dev_out + M, dev_qs0 + M, dev_qs1 + M, dev_expWds2_m1, dev_expWds_m1, M);
        } else {
            Richardson << <(M + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> >(dev_out + M, dev_qs0 + M, dev_qs1 + M, dev_expWds2_p1, dev_expWds_p1, M);
        }
    }
    
}; // end of step class


class density
//------------------------------------------------------------
// evaluates phi- and phi+ stored at W+2*M and W+3*M
//
// propagators are accessed with either 
// q1[s][r]=qq1[s*M+r] = q(r,s)
// q2[s][r]=qq2[s*M+r] = q^dagger(r,s)
//------------------------------------------------------------
{
    const int Ns;
    
    class step *S1;
    
    // DEVICE MEMORY FOR W-, W+, rA and rB //
    double *dev_W;
    
    //Cube fit stuff
    double *dev_cubes, *dev_cubesT, *dev_x;
    
    // DEVICE MEMORY FOR PROPAGATORS //
    double **dev_q1, **dev_q2, *dev_qq;
    
    // DEVICE MEMORY TO EXPRESS DENSITY AS rA-rB and rA+rB-1 //
    double *dev_A, *dev_B;
    
    // MINIMUM NUMBER OF BLOCKS REQUIRED TO COVER M MESH POINTS WITH threadsPerBlock //
    int nblocks;
    
    // HOST MEMORY TO HOLD RESULTS FROM REDUCTIONS //
    double *reduced;
    
    // DEVICE MEMORY TO HOLD RESULTS FROM REDUCTIONS //
    double *dev_reduced;
    
    // BATCH SIZE //	
    int _batch = 2;
    
public:
    //------------------------------------------------------------
    
    density(const int Ns, double *DA):Ns(Ns)
    {
        S1 = new step(DA);
        
        // ALLOCATE DEVICE MEMORY FOR W-, W+, rA and rB //
        HANDLE_ERROR(cudaMalloc((void**)&dev_W, 14*M*sizeof(double))); //unnecessarily large for now
        HANDLE_ERROR(cudaMalloc((void**)&dev_cubes, 4*strn*M*sizeof(double)));
        HANDLE_ERROR(cudaMalloc((void**)&dev_cubesT, 5*strn*M*sizeof(double)));
        HANDLE_ERROR(cudaMalloc((void**)&dev_x, strn*sizeof(double)));
        
        
        // ALLOCATE DEVICE MEMORY FOR PROPAGATORS //
        HANDLE_ERROR(cudaMalloc((void**)&dev_qq, _batch*(Ns + 1)*M * sizeof(double)));
        
        // CREATE POINTERS TO q1 AND q2 CONTIGIOUS MEMORY FOR CUFFTPLANMANY() //
        dev_q1 = new double *[Ns + 1];
        dev_q2 = new double *[Ns + 1];
        for (int i = 0; i < Ns + 1; i++) {
            dev_q1[i] = dev_qq + 2 * i*M;
            dev_q2[i] = dev_qq + (2 * i + 1)*M;
        }
        
        // ALLOCATE DEVICE MEMORY TO EXPRESS DENSITY AS rA-rB AND rA+rB-1 //
        HANDLE_ERROR(cudaMalloc((void**)&dev_A, M * sizeof(double)));
        HANDLE_ERROR(cudaMalloc((void**)&dev_B, M * sizeof(double)));
        
        // DETERMINE NUMBER OF BLOCKS REQUIRED WITH CHOSEN threadsPerBlock //
        nblocks = (M + threadsPerBlock - 1) / threadsPerBlock;
        
        // ALLOCATE ARRAY TO HOLD RESULTS FROM GPU REDUCTIONS //
        reduced = new double[nblocks];
        
        // ALLOCATE DEVICE MEMORY TO HOLD RESULTS FROM GPU REDUCTIONS //
        HANDLE_ERROR(cudaMalloc((void**)&dev_reduced, nblocks * sizeof(double)));
    }
    
    //------------------------------------------------------------
    
    ~density()
    {
        delete S1;
        delete[] dev_q1, dev_q2, reduced;
        HANDLE_ERROR(cudaFree(dev_W));
        HANDLE_ERROR(cudaFree(dev_qq));
        HANDLE_ERROR(cudaFree(dev_A));
        HANDLE_ERROR(cudaFree(dev_B));
        HANDLE_ERROR(cudaFree(dev_reduced));
        
        HANDLE_ERROR(cudaFree(dev_cubes));
        HANDLE_ERROR(cudaFree(dev_cubesT));
        HANDLE_ERROR(cudaFree(dev_x));
    }
    
    //------------------------------------------------------------
    
    void props (double *w, double *lnQ, int ii, int Ns, double alpha, double phic, double * phis, double f)
    {
        const int n1=(int) round (Ns*f), n2=Ns-n1, Nh = alpha * Ns;
        const double ds=1.0/Ns;
        f=double(n1)/Ns;
        
        double Q1=0, Q2=0, QC=0;
        int    s, wt;
        double zA = 0, zB, zC;
        
        double *phiA=w+2*M;
        
        //device pointers
        double *dev_rA=dev_W+2*M, *dev_rB=dev_W+3*M,  *dev_rh=dev_W+4*M, *dev_rCB=dev_W+5*M, *dev_rhA=dev_W+6*M, *dev_rP=dev_W+7*M, *dev_rM=dev_W+8*M, *dev_rCA=dev_W+9*M,*dev_rC=dev_W+10*M, *dev_rhB=dev_W+11*M;
        
        double ptt=0;
        
        // COPY THE W- AND W+ FIELDS TO THE DEVICE //
        HANDLE_ERROR(cudaMemcpy(dev_W, w, 2 * M * sizeof(double), cudaMemcpyHostToDevice));
        
        /* Laplacian part of diffusion equation */
        S1->prepareFactors(ds, dev_W);
        
        /* Calculate propagators: q1 and q2 */
        Array_set_value << <(M + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> >(dev_q1[0], 1.0, M);
        Array_set_value << <(M + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> >(dev_q2[0], 1.0, M);
        
        
        
        // USE CUFFTPLANMANY() TO GET q(s+1) FROM q(s) //
        for (int s = 0; s < Ns; s++) S1->do_step(dev_q1[s], dev_q1[s + 1], s, n1, n2);
        
        /* Calculate total partition functions: Q1, Q2 and QC */
        reduction_sum<<<1, threadsPerBlock>>>(dev_q2[Nh], dev_reduced, M);
        HANDLE_ERROR(cudaMemcpy(&Q2, dev_reduced, 1 * sizeof(double), cudaMemcpyDeviceToHost));
        
        reduction_sum<<<1, threadsPerBlock>>>(dev_q1[Ns], dev_reduced, M);
        HANDLE_ERROR(cudaMemcpy(&QC, dev_reduced, 1 * sizeof(double), cudaMemcpyDeviceToHost));
        
        
        Q1 /= M;
        Q2 /= M;
        QC /= M;
        
        //CAN:
        //use this to run in the canonical ensemble. phic then represents the volume fracrtion of copolymers
        /*
        *lnQ = phic*log(QC) + ((1.0-phic)/alpha)*log(Q2); //CAN
        zA=0;
        zB = (1.0-phic)/(Q2*alpha);
        zC = phic/QC;
         */
        
        //GC:
        //use this to run in the semi-grand canonical ensemble. Here phic represents the fugacity (e^mu) of the copolymers
        zA=0;
        zB = 1.0;
        zC = phic;
        *lnQ = zA*Q1 + zB*Q2 + zC*QC;//GC //zs?
        
        
        //end concentrations
        rA_rB_init_Ns_homo << <(M + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (dev_rhA, dev_rhB, dev_rCA, dev_rCB,  dev_q1[Nh], dev_q2[Nh], dev_q1[Ns], dev_q2[Ns], dev_q1[n1], dev_q2[n2], M);
        
        //middle segments in homopolymer A
        for (s = 1, wt = 4; s < Nh; s++, wt = 6 - wt) {
            rA_rB_mid_segments<<<(M+threadsPerBlock-1)/threadsPerBlock, threadsPerBlock>>>(dev_rhA, dev_q1[s], dev_q1[Nh - s], wt, M);
        }
        
        //middle segments in homopolymer B
        for (s = 1, wt = 4; s < Nh; s++, wt = 6 - wt) {
            rA_rB_mid_segments<<<(M+threadsPerBlock-1)/threadsPerBlock, threadsPerBlock>>>(dev_rhB, dev_q2[s], dev_q2[Nh - s], wt, M);
        }
        
        //middle segments in copolymer block A
        for (s = 1, wt = 4; s < n1; s++, wt = 6 - wt) {
            rA_rB_mid_segments<<<(M+threadsPerBlock-1)/threadsPerBlock, threadsPerBlock>>>(dev_rCA, dev_q1[s], dev_q2[Ns - s], wt, M);
        }
        
        //middle segments in copolymer block B
        for (s = n1+1, wt = 4; s < Ns; s++, wt = 6 - wt) {
            rA_rB_mid_segments<<<(M+threadsPerBlock-1)/threadsPerBlock, threadsPerBlock>>>(dev_rCB, dev_q1[s], dev_q2[Ns - s], wt, M);
        }
        
        //scale concentrations
        scale_concentrations<<<(M+threadsPerBlock-1)/threadsPerBlock, threadsPerBlock>>>(dev_rhA, dev_rhB, dev_rCA, dev_rCB, zA, zB, zC, ds, ds, M);
        
        
        
        //combine concentrations (rA += rCA, rB+=rCB, rC=rCA+rCB)
        sumconcs2<<<(M+threadsPerBlock-1)/threadsPerBlock, threadsPerBlock>>>(dev_rA, dev_rB, dev_rh, dev_rCB, dev_rhA, dev_rhB, dev_rP, dev_rM, dev_rCA, dev_rC, M);
        
        //calculate averages of homopolymer concentrations
        reduction_sum<<<1, threadsPerBlock>>>(dev_rA, dev_reduced, M);
        HANDLE_ERROR(cudaMemcpy(&ptt, dev_reduced, 1 * sizeof(double), cudaMemcpyDeviceToHost));
        phis[1]=ptt;//pt;
        
        reduction_sum<<<1, threadsPerBlock>>>(dev_rB, dev_reduced, M);
        HANDLE_ERROR(cudaMemcpy(&ptt, dev_reduced, 1 * sizeof(double), cudaMemcpyDeviceToHost));
        phis[2]=ptt;
        
        //calculate averages of homopolymer concentrations
        reduction_sum<<<1, threadsPerBlock>>>(dev_rC, dev_reduced, M);
        HANDLE_ERROR(cudaMemcpy(&ptt, dev_reduced, 1 * sizeof(double), cudaMemcpyDeviceToHost));
        phis[0]=ptt;
        
        reduction_sum<<<1, threadsPerBlock>>>(dev_rhA, dev_reduced, M);
        HANDLE_ERROR(cudaMemcpy(&ptt, dev_reduced, 1 * sizeof(double), cudaMemcpyDeviceToHost));
        phis[3]=ptt;
        
        reduction_sum<<<1, threadsPerBlock>>>(dev_rhB, dev_reduced, M);
        HANDLE_ERROR(cudaMemcpy(&ptt, dev_reduced, 1 * sizeof(double), cudaMemcpyDeviceToHost));
        phis[4]=ptt;
        
        phis[0] /= M; phis[1] /= M; phis[2] /= M; phis[3] /= M; phis[4] /= M;
        
        //send stuff to host array
        HANDLE_ERROR(cudaMemcpy(phiA, dev_rA, 10 * M * sizeof(double), cudaMemcpyDeviceToHost));
        
    }
    
    //------------------------------------------------------------
    
    void cubefit2 (double *x, double **ww)
    {
        for(int i=0;i<strn;i++){
            HANDLE_ERROR(cudaMemcpy(dev_W, ww[i], 2*M * sizeof(double), cudaMemcpyHostToDevice));
            //transpose data to make it easier to handle/fit
            transpose_cube<<<(M+threadsPerBlock-1)/threadsPerBlock, threadsPerBlock>>>(dev_W, dev_cubes, i, M,strn);
        }
        HANDLE_ERROR(cudaMemcpy(dev_x, x, strn * sizeof(double), cudaMemcpyHostToDevice));
        //calculate spline coefficients
        cubefit_g<<<(M+threadsPerBlock-1)/threadsPerBlock, threadsPerBlock>>>(dev_x, dev_cubes, dev_cubesT, M);
        cudaDeviceSynchronize();
        HANDLE_ERROR(cudaMemcpy(cubes, dev_cubes, 4*strn*M * sizeof(double), cudaMemcpyDeviceToHost));
    }
    
}; // end of density class
