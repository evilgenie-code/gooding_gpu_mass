#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define _CRT_SECURE_NO_WARNINGS

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h> 

#include <time.h>

using namespace std;

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))


__device__ void getUnit(double* r, double R, double* Unit)
{
    for (int j = 0; j < 3; j++)
    {
        Unit[j] = r[j] / R;
    }
}

__device__ void getUnitDouble(double* r, double R, double* Unit)
{
    int j;

    for (j = 0; j < 3; j++)
    {
        Unit[j] = r[j] / R;
    }
}

__device__ double getVetVal(double* vet1)
{
    double temp = 0.0;

    for (int i = 0; i < 3; i++)
    {
        temp += vet1[i] * vet1[i];
    }

    return sqrt(temp);
}

__device__ void cross(double* vet1, double* vet2, double* prod)
{
    prod[0] = (vet1[1] * vet2[2] - vet1[2] * vet2[1]);
    prod[1] = (vet1[2] * vet2[0] - vet1[0] * vet2[2]);
    prod[2] = (vet1[0] * vet2[1] - vet1[1] * vet2[0]);
}

__device__ void sigmax(double y, double sig = 0.0, double dsigdx = 0.0, double d2sigdx2 = 0.0, double d3sigdx3 = 0.0)
{
    double powers[25];
    int i;
    // preload the factors[an]
    // (25 factors is more than enough for 16 - digit accuracy)
    double an[25] = { 4.000000000000000e-001,     2.142857142857143e-001,     4.629629629629630e-002,
                      6.628787878787879e-003,     7.211538461538461e-004,     6.365740740740740e-005,
                      4.741479925303455e-006,     3.059406328320802e-007,     1.742836409255060e-008,
                      8.892477331109578e-010,     4.110111531986532e-011,     1.736709384841458e-012,
                      6.759767240041426e-014,     2.439123386614026e-015,     8.203411614538007e-017,
                      2.583771576869575e-018,     7.652331327976716e-020,     2.138860629743989e-021,
                      5.659959451165552e-023,     1.422104833817366e-024,     3.401398483272306e-026,
                      7.762544304774155e-028,     1.693916882090479e-029,     3.541295006766860e-031,
                      7.105336187804402e-033 };

    for (i = 0; i < 25; i++)
    {
        //powers of y
        powers[i] = pow(y, i + 1);
        sig += 4 / 3 + powers[i] * an[i];
    }

    for (i = 0; i < 24; i++)
    {
        // dsigma / dx
        dsigdx += ((i + 2) * powers[i + 1]) * an[i + 1];
    }

    for (i = 0; i < 23; i++)
    {
        // d2sigma / dx2
        d2sigdx2 = ((i + 3) * (i + 2) * powers[i + 2]) * an[i + 2];
    }

    for (i = 0; i < 23; i++)
    {
        // d3sigma / dx3
        d3sigdx3 = ((i + 4) * (i + 3) * (i + 2) * powers[i + 3]) * an[i + 3];
    }

    dsigdx = dsigdx + an[0];
    d2sigdx2 = d2sigdx2 + 0 + 2 * an[1];
    d3sigdx3 = d3sigdx3 + 0 + 0 + 3 * 2 * an[2];

}

__device__ void LancasterBlanchard(double x, double q, double m, double* Tprod)
{
    const double M_PI = 3.14159265358979323846264338327950288419716939937510;

    double E, sig1 = 0.0, sig2 = 0.0, dsigdx1 = 0.0,
        dsigdx2 = 0.0, d2sigdx21 = 0.0, d2sigdx22 = 0.0, d3sigdx31 = 0.0, d3sigdx32 = 0.0,
        y, z, f, g, d;
    // protection against idiotic input
    if (x < -1) //impossible; negative eccentricity
    {
        x = abs(x) - 2;
    }
    if (x == -1) // impossible; offset x slightly
    {
        x = x + 0.00001;
    }

    // compute parameter E
    E = x * x - 1;

    // T(x), T'(x), T''(x)
    if (x == 1) // exactly parabolic; solutions known exactly
    {
        Tprod[0] = 4. / 3 * (1 - pow(q, 3));
        Tprod[1] = 4. / 5 * (pow(q, 5) - 1);
        Tprod[2] = Tprod[1] + 120 / 70 * (1 - pow(q, 7));
        Tprod[3] = 3 * (Tprod[2] - Tprod[1]) + 2400 / 1080 * (pow(q, 9) - 1);
    }
    else if (fabs(x - 1) < 1e-2) // near - parabolic; compute with series
                                 // evaluate sigma
    {
        sigmax(-E, sig1, dsigdx1, d2sigdx21, d3sigdx31);
        sigmax(-E * q * q, sig2, dsigdx2, d2sigdx22, d3sigdx32);
        Tprod[0] = sig1 - pow(q, 3) * sig2;
        Tprod[1] = 2. * x * (pow(q, 5) * dsigdx2 - dsigdx1);
        Tprod[2] = Tprod[1] / x + 4 * pow(x, 2) * (d2sigdx21 - pow(q, 7) * d2sigdx22);
        Tprod[3] = 3 * (Tprod[2] - Tprod[1] / x) / x + 8. * x * x * (pow(q, 9) * d3sigdx32 - d3sigdx31);
    }
    else // all other cases
    {
        // compute all substitution functions
        y = sqrt(fabs(E));
        z = sqrt(1 + pow(q, 2) * E);
        f = y * (z - q * x);
        g = x * z - q * E;
        d = (E < 0) * (atan2(f, g) + M_PI * m) + (E > 0) * logf(MAX(0, f + g));
        Tprod[0] = 2 * (x - q * z - d / y) / E;
        Tprod[1] = (4. - 4. * pow(q, 3) * x / z - 3 * x * Tprod[0]) / E;
        Tprod[2] = (-4. * pow(q, 3) / z * (1 - pow(q, 2) * pow(x, 2) / pow(z, 2)) - 3 * Tprod[0] - 3 * x * Tprod[1]) / E;
        Tprod[3] = (4. * pow(q, 3) / pow(z, 2) * ((1 - pow(q, 2) * pow(x, 2) / pow(z, 2)) + 2 * pow(q, 2) * x / pow(z, 2) * (z - x)) - 8 * Tprod[1] - 7 * x * Tprod[2]) / E;

    }
}

__device__ double getXM(double phr, int revs)
{
    const double M_PI = 3.14159265358979323846264338327950288419716939937510;

    double xMpi = 4 / (3 * M_PI * (2 * revs + 1));

    if (phr < M_PI)
    {
        return xMpi * pow(phr / M_PI, 0.125);
    }
    else if (phr > M_PI)
    {
        return xMpi * (2.0 - pow(2 - (phr / M_PI), 0.125));
    }

    return 0;

}

__device__ double halleysMethod(double xM, double q, int revs)
{
    int iterations = 0;
    const double tol = 1e-11;
    double tProd[4] = { 0.0, 0.0, 0.0, 0.0 }, Tp = 0, Tpp, Tppp, xMp;

    while (fabs(Tp) > tol)
    {
        iterations = iterations + 1;

        LancasterBlanchard(xM, q, revs, tProd);

        Tp = tProd[1];
        Tpp = tProd[2];
        Tppp = tProd[3];

        xMp = xM;
        xM = xM - 2. * Tp * Tpp / (2 * pow(Tpp, 2) - Tp * Tppp);

        if (iterations % 7 == 0)
        {
            xM = (xMp + xM) / 2;
        }
        //
        //        if (iterations > 25)
        //        {
        //            return ;
        //        }
    }

    return xM;
}

__global__ void lambert(const float* r1, const float* r2, float* dt, int revs, int lw, float mu, float* v1, float* v2)
{
    const double M_PI = 3.14159265358979323846264338327950288419716939937510;


    int j, leftbranch = (mu > 0) - (mu < 0);
    double R1 = 0.0, R2 = 0.0, dotProd = 0.0, mcrsProd, theta, c, s, T, q,
        T0 = 0.0, dummy = 0.0;
    double  r1Unit[3], r2Unit[3], crsProd[3], ucrsProd[3], th1Unit[3], th2Unit[3],
        r1Double[3], r2Double[3], Tprod0[4] = { T0, 0.0, 0.0, 0.0 }, Td, phr, Tprod[4];

    int blockIndex = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x;
    int ThreadIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
    int tid = blockIndex * blockDim.x * blockDim.y * blockDim.z + ThreadIndex;
    double t = dt[tid];


    int off = 3;

    const double tol = 1.e-15;

    if (t <= 0)
    {
        //printf("ERROR in Lambert Solver: Negative Time in input.\n");
        return;
    }

    for (j = 0; j < 3; j++)
    {
        r1Double[j] = r1[tid * off + j];
        r2Double[j] = r2[tid * off + j];
    }

    for (j = 0; j < 3; j++)
    {
        R1 += pow(r1Double[j], 2);
        R2 += pow(r2Double[j], 2);
    }

    R1 = sqrt(R1);
    R2 = sqrt(R2);

    getUnit(r1Double, R1, r1Unit);
    getUnit(r2Double, R2, r2Unit);

    cross(r1Double, r2Double, crsProd); // перекрестное произведение

    mcrsProd = getVetVal(crsProd); //Величина этого перекрестного произведений

    getUnitDouble(crsProd, mcrsProd, ucrsProd);

    cross(ucrsProd, r1Unit, th1Unit);
    cross(ucrsProd, r2Unit, th2Unit);

    for (j = 0; j < 3; j++)
    {
        dotProd += (r1Double[j] * r2Double[j]);
    }

    theta = acos(MAX(-1, MIN(1, dotProd / (R1 * R2))));

    if (lw == 1)
    {
        theta = theta - 2 * M_PI;
    }

    c = sqrt(pow(R1, 2) + pow(R2, 2) - 2.0 * R1 * R2 * cos(theta));
    s = (R1 + R2 + c) / 2;
    T = sqrt(8.0 * mu / pow(s, 3)) * t;
    q = (sqrt(R1 * R2) / s) * cos(theta / 2);

    LancasterBlanchard(0, q, revs, Tprod0);

    T0 = Tprod0[0];
    Td = T0 - T;
    phr = fmod(2.0 * atan2(1.0 - pow(q, 2), 2.0 * q), 2.0 * M_PI);

    // initial output is pessimistic
    for (int i = 0; i < 3; i++)
    {
        v1[i] = 0.0;
        v2[i] = 0.0;
    }

    double x01;
    double x0, x02, x03, W, lambda, xMpi, xM, Tp;

    if (revs == 0)
    {
        x01 = T0 * Td / 4 / T;

        if (Td > 0)
        {
            x0 = x01;
        }
        else
        {
            x01 = Td / (4. - Td);
            x02 = -sqrt(-Td / (T + T0 / 2));
            W = x01 + 1.7 * sqrt(2 - phr / M_PI);

            if (W >= 0)
            {
                x03 = x01;
            }
            else
            {
                x03 = x01 + pow(-W, (1.0 / 16)) * (x02 - x01);
            }

            lambda = 1. + x03 * (1. + x01) / 2. - 0.03 * pow(x03, 2) * sqrt(1 + x01);
            x0 = lambda * x03;
        }

        if (x0 < -1)
        {
            return;
        }
    }
    else
    {
        //multi-rev cases
        // determine minimum Tp(x)
        xM = getXM(phr, revs);
        Tp = 1.0;

        // use Halley's method
        xM = halleysMethod(xM, q, revs);

        // xM should be elliptic(-1 < x < 1)
        // (this should be impossible to go wrong)
        if ((xM < -1) || (xM > 1))
        {
            return;
        }

        //corresponding time
        LancasterBlanchard(xM, q, revs, Tprod);

        double TM = Tprod[0];
        double Tpp;

        // T should lie above the minimum T
        if (TM > t)
        {
            return;
        }

        //find two initial values for second solution(again with lambda - type patch)

        // some initial values


        double TmTM = T - TM;
        double  T0mTM = T0 - TM;

        LancasterBlanchard(xM, q, revs, Tprod);
        Tpp = Tprod[2];

        if (leftbranch > 0)
        {
            double x = sqrt(TmTM / (Tpp / 2 + TmTM / pow((1 - xM), 2)));
            W = xM + x;
            W = 4.0 * W / (4.0 + TmTM) + pow((1 - W), 2);
            x0 = x * (1 - (1 + revs + (theta - 0.5)) / (1 + 0.15 * revs) * x * (W * 0.5 + 0.03 * x * sqrt(W))) + xM;

            if (x0 > 1)
            {
                return;
            }
        }
        else
        {
            if (Td > 0)
            {
                x0 = xM - sqrt(TM / (Tpp / 2 - TmTM * (Tpp / 2 / T0mTM - 1 / pow(xM, 2))));
            }
            else
            {
                double x00 = Td / (4.0 - Td);
                W = x00 + 1.7 * sqrt(2 * (1 - phr));
                if (W >= 0)
                {
                    x03 = x00;
                }
                else
                {
                    x03 = x00 - sqrt(pow((-W), 0.125) * (x00 + sqrt(-Td / (1.5 * T0 - Td))));
                }

                W = 4 / (4 - Td);
                lambda = (1 + (1 + revs + 0.24 * (theta - 0.5)) /
                    ((1 + 0.15 * revs) * x03 * (0.5 * W - 0.03 * x03 * sqrt(W))));
                x0 = x03 * lambda;
            }
        }

        if (x0 < -1)
        {
            return;
        }
    }

    //find root of Lancaster & Blancard's function
    //(Halley's method)
    //printf("%f \n", x0);
    double x = x0;
    double Tx = 99999;
    int iterations = 0;

    while (std::abs(Tx) > tol)
    {
        iterations = iterations + 1;
        // compute function value, and first two derivatives

        LancasterBlanchard(x, q, revs, Tprod);
        Tx = Tprod[0];
        Tp = Tprod[1];
        double Tpp = Tprod[2];

        // find the root of the *difference* between the
        // function value[T_x] and the required time[T]
        Tx = Tx - T;

        // new value of x
        double xp = x;
        x = x - 2. * Tx * Tp / (2. * pow(Tp, 2) - Tx * Tpp);
        if (iterations % 7 == 0)
        {
            x = (xp + x) / 2.;
        }

        if (iterations > 250)
        {
            return;
        }
    }


    double gamma = sqrt(mu * s / 2.);

    double sigma, rho, z;

    if (c == 0)
    {
        sigma = 1;
        rho = 0;
        z = std::abs(x);
    }
    else
    {
        sigma = 2 * sqrt(R1 * R2 / pow(c, 2)) * sin(theta / 2);
        rho = (R1 - R2) / c;
        z = sqrt(1.0 + (pow(q, 2) * (pow(x, 2) - 1)));
    }


    // radial component

    double Vr1 = +gamma * ((q * z - x) - rho * (q * z + x)) / R1;
    double Vr2 = -gamma * ((q * z - x) + rho * (q * z + x)) / R2;

    //tangential component

    double Vtan1 = sigma * gamma * (z + q * x) / R1;
    double Vtan2 = sigma * gamma * (z + q * x) / R2;

    double Vr1vec[3], Vr2vec[3], Vtan1vec[3], Vtan2vec[3];

    for (j = 0; j < 3; j++)
    {
        Vr1vec[j] = Vr1 * r1Unit[j];
        Vr2vec[j] = Vr2 * r2Unit[j];
        Vtan1vec[j] = Vtan1 * th1Unit[j];
        Vtan2vec[j] = Vtan2 * th2Unit[j];

        //Cartesian velocity
        v1[tid * off + j] = Vtan1vec[j] + Vr1vec[j];
        v2[tid * off + j] = Vtan2vec[j] + Vr2vec[j];

        //printf("v1[%i] = %f, v2[%i] = %f \n", j, v1[j], j, v2[j]);
    }

}

double** reading_data(char* name_file, double** DATA, char ADD[][1000], int ignore, int& SIZE1, int SIZE2) {

    SIZE1 = 1;
    int i = 0, j;
    DATA = (double**)malloc(SIZE1 * sizeof(double*));

    char line[1000], * tok, * next_token = NULL;


    ifstream file(name_file);
    while (file.getline(line, 1000) && ignore) {
        ignore--;
        strcpy(ADD[i], line);
        i++;
    }
    i = SIZE1 - 1;
    do {

        DATA = (double**)realloc(DATA, ++SIZE1 * sizeof(double*));
        DATA[i] = (double*)malloc(SIZE2 * sizeof(double));

        for (char* tok = strtok_s(line, " ", &next_token), j = 0; tok; tok = strtok_s(NULL, " ", &next_token)) {

            DATA[i][j] = atof(tok);
            j++;
            if (j == SIZE2) break;
        }

        i++;
        //if (i==3000) break;      	

    } while (file.getline(line, 1000));
    file.close();


    return DATA;
}

void writing_data(char* name_file, double** DATA, int SIZE1, int SIZE2, char ADD[][1000], int add) {

    int i, j;
    FILE* fileout;
    fileout = fopen(name_file, "w");

    for (i = 0; i < add; i++) fprintf(fileout, "%s\n", ADD[i]);

    for (i = 0; i < SIZE1 - 1; i++) {
        for (j = 0; j < SIZE2; j++)
            fprintf(fileout, "%26.16e", DATA[i][j]);
        fprintf(fileout, "\n");
    }
    fclose(fileout);
}

int div_up(int x, int y)
{
    return (x - 1) / y + 1;
}

void vett_cpu(const double* vet1, const double* vet2, double* prod)
{
    prod[0] = (vet1[1] * vet2[2] - vet1[2] * vet2[1]);
    prod[1] = (vet1[2] * vet2[0] - vet1[0] * vet2[2]);
    prod[2] = (vet1[0] * vet2[1] - vet1[1] * vet2[0]);
}

int main() {

    cudaSetDevice(1);
    int threads = 384;
    int blocs;
    int leght = 3;

    printf("1. start program\n");

    double AU = 1.49597870691e8;
    double fMSun = 1.32712440018e11;             // km^3/sec^2

    double UnitR = AU;
    double UnitV = sqrt(fMSun / UnitR);          // km/sec
    double UnitT = (UnitR / UnitV) / 86400;      // day

    double mu = 1.;							// гравитационная постоянная
    int nrev = 0;							// число витков
    int lw = -1;

    double dv1[3], dv2[3], dV1, dV2;

    char name_file[] = { "data1.txt" };
    char name_file2[] = { "data1_izzo_cpu.txt" };
    double** DATA = NULL;
    int i, k, SIZE1, SIZE2 = 29;
    char boof[2][1000];
    double R0[3] = { 0.0,0.0,0.0 };

    printf("2. reading file \n");

    DATA = reading_data(name_file, DATA, boof, 2, SIZE1, SIZE2);

    printf("3. finish reading file\n");
    printf("4. count tasks %i \n", SIZE1);
    printf("5. start calculate \n");

    blocs = div_up(SIZE1, threads);
    const int countTasks = SIZE1;


    printf("6. Blocs = %i, Threads = %i \n", blocs, threads);

    int sizeCudaVariableBig = (SIZE1 - 1) * 3;
    int sizeCudaVariableSmall = (SIZE1 - 1);

    float* r0 = new float[sizeCudaVariableBig];
    float* r1 = new float[sizeCudaVariableBig];
    float* v1 = new float[sizeCudaVariableBig];
    float* v2 = new float[sizeCudaVariableBig];
    float* dt = new float[sizeCudaVariableSmall];
    int* lww = new int[sizeCudaVariableSmall];
    int lenght = 3;
    int off = 3;

    float* dev_r0, * dev_r1, * dev_v1, * dev_v2, * dev_dt;
    int* dev_lw;

    for (int n = 0; n < SIZE1 - 1; n++)
    {
        dt[n] = DATA[n][14] / UnitT;

        vett_cpu(&DATA[n][0], &DATA[n][6], R0);
        if (R0[2] >= 0.0) lww[n] = 0;
        else lww[n] = 1;

        for (int m = 0; m < lenght; m++)
        {
            r0[n * off + m] = DATA[n][m];
            r1[n * off + m] = DATA[n][6 + m];
        }
    }

    cudaMalloc((void**)&dev_r0, sizeCudaVariableBig * sizeof(float));
    cudaMalloc((void**)&dev_r1, sizeCudaVariableBig * sizeof(float));
    cudaMalloc((void**)&dev_v1, sizeCudaVariableBig * sizeof(float));
    cudaMalloc((void**)&dev_v2, sizeCudaVariableBig * sizeof(float));
    cudaMalloc((void**)&dev_dt, sizeCudaVariableSmall * sizeof(float));
    cudaMalloc((void**)&dev_lw, sizeCudaVariableSmall * sizeof(int));

    cudaMemcpy(dev_r0, r0, sizeCudaVariableBig * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_r1, r1, sizeCudaVariableBig * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dt, dt, sizeCudaVariableSmall * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_lw, lww, sizeCudaVariableSmall * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float gpuTime = 0.0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    lambert << <blocs, threads >> > (dev_r0, dev_r1, dev_dt, 1, 0, 1., dev_v1, dev_v2);

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpuTime, start, stop);

    printf("7. Time one on GPU = %16.10e miliseconds\n", (gpuTime / (SIZE1 - 1)));
    printf("7.1. Time on GPU = %16.10e miliseconds\n", gpuTime);

    cudaMemcpy(v1, dev_v1, sizeCudaVariableBig * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v2, dev_v2, sizeCudaVariableBig * sizeof(float), cudaMemcpyDeviceToHost);

    printf("8. finish calculate \n");

    for (int n = 0; n < SIZE1 - 1; n++)
    {
        for (int m = 0; m < lenght; m++)
        {
            r0[n * off + m] = DATA[n][m];
            r1[n * off + m] = DATA[n][6 + m];
        }
    }

    for (i = 0; i < SIZE1 - 1; i++) {

        dV1 = 0; dV2 = 0;
        for (k = 0; k < 3; k++) {
            dv1[k] = DATA[i][18 + k] - v1[i * off + k];
            dv2[k] = DATA[i][21 + k] - v2[i * off + k];
            dV1 += dv1[k] * dv1[k];
            dV2 += dv2[k] * dv2[k];
        }

        dV1 = sqrt(dV1) * UnitV;
        dV2 = sqrt(dV2) * UnitV;

        DATA[i][SIZE2 - 2] = DATA[i][15] - dV1;
        DATA[i][SIZE2 - 1] = DATA[i][16] - dV2;

        DATA[i][15] = dV1;
        DATA[i][16] = dV2;
        DATA[i][17] = DATA[i][15] + DATA[i][16];

    }

    printf("9. start write data \n");

    writing_data(name_file2, DATA, SIZE1, SIZE2, boof, 2);

    printf("10. finish write data \n");

    cudaFree(dev_r0);
    cudaFree(dev_r1);
    cudaFree(dev_v1);
    cudaFree(dev_v2);
    cudaFree(dev_lw);
    cudaFree(dev_dt);


    printf("11. finish program");

    return 0;
}