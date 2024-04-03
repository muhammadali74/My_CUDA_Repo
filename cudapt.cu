#include <math.h>   // smallpt, a Path Tracer by Kevin Beason, 2008
#include <stdlib.h> // Make : g++ -O3 -fopenmp smallpt.cpp -o smallpt
#include <stdio.h>  //        Remove "-fopenmp" for g++ version < 4.2
#include <random>

#include <iostream>
#include <cuda_runtime.h>
#include <vector_types.h>
#include "device_launch_parameters.h"
#include <vecutils.h> // from http://www.icmc.usp.br/~castelo/CUDA/common/inc/cutil_math.h
#include <curand_kernel.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// std::default_random_engine generator;
// std::uniform_real_distribution<double> distr(0.0, 1.0);
// double curand_uniform(unsigned short *X)
// {
//     return distr(generator);
// }
// float3 operator+(const float3 &b) { return Vec(x + b.x, y + b.y, z + b.z); }
// float3 operator-(const Vec &b) { return Vec(x - b.x, y - b.y, z - b.z); }
// float3 operator*(double b)  { return Vec(x * b, y * b, z * b); }
float3 mult(const float3 &a, const float3 &b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
// float3 &norm(float3 a) { return *a = *a * (1 / sqrt(a.x * a.x + y * y + z * z)); }
// double dot(const Vec &b) { return x * b.x + y * b.y + z * b.z; } // cross:
float3 modd(float3 &a, float3 &b) { return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }

struct Ray
{
    float3 o, d;
    __device__ Ray(float3 o_, float3 d_) : o(o_), d(d_) {}
};
enum Refl_t
{
    DIFF,
    SPEC,
    REFR
}; // material types, used in radiance()

struct Sphere
{
    double rad;     // radius
    float3 p, e, c; // position, emission, color
    Refl_t refl;    // reflection type (DIFFuse, SPECular, REFRactive)
    Sphere(double rad_, float3 p_, float3 e_, float3 c_, Refl_t refl_) : rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {}
    __device__ double intersect(const Ray &r) const
    {                        // returns distance, 0 if nohit
        float3 op = p - r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
        double t, eps = 1e-4, b = dot(op, r.d), det = b * b - dot(op, op) + rad * rad;
        if (det < 0)
            return 0;
        else
            det = sqrt(det);
        return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
    }
};

__constant__ Sphere spheres[] = {
    // Scene: radius, position, emission, color, material
    Sphere(1e5, make_float3(1e5 + 1, 40.8, 81.6), float3(), make_float3(.75, .25, .25), DIFF),   // Left
    Sphere(1e5, make_float3(-1e5 + 99, 40.8, 81.6), float3(), make_float3(.25, .25, .75), DIFF), // Rght
    Sphere(1e5, make_float3(50, 40.8, 1e5), float3(), make_float3(.75, .75, .75), DIFF),         // Back
    Sphere(1e5, make_float3(50, 40.8, -1e5 + 170), float3(), float3(), DIFF),                    // Frnt
    Sphere(1e5, make_float3(50, 1e5, 81.6), float3(), make_float3(.75, .75, .75), DIFF),         // Botm
    Sphere(1e5, make_float3(50, -1e5 + 81.6, 81.6), float3(), make_float3(.75, .75, .75), DIFF), // Top
    Sphere(16.5, make_float3(27, 16.5, 47), float3(), make_float3(1, 1, 1) * .999, SPEC),        // Mirr
    Sphere(16.5, make_float3(73, 16.5, 78), float3(), make_float3(1, 1, 1) * .999, REFR),        // Glas
    Sphere(600, make_float3(50, 681.6 - .27, 81.6), make_float3(12, 12, 12), float3(), DIFF)     // Lite
};
inline __host__ __device__ double clamp(double x) { return x < 0 ? 0 : x > 1 ? 1
                                                                             : x; }

inline __host__ __device__ inline int toInt(double x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }

inline __host__ __device__ bool intersect(const Ray &r, double &t, int &id) // all args device frinedly
{
    double n = sizeof(spheres) / sizeof(Sphere), d, inf = t = 1e20;
    for (int i = int(n); i--;)
        if ((d = spheres[i].intersect(r)) && d < t) // data dependency so cant paralleize. divergence ka boht khtra
        {
            t = d;
            id = i;
        }
    return t < inf;
}

__device__ float3 radiance(const Ray &r, int depth, curandState *state)
{
    double t;   // distance to intersection
    int id = 0; // id of intersected object
    if (!intersect(r, t, id))
        return float3();             // if miss, return black
    const Sphere &obj = spheres[id]; // the hit object
    float3 x = r.o + r.d * t, n = normalize(x - obj.p), nl = dot(n, r.d) < 0 ? n : n * -1, f = obj.c;
    double p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y
                                                        : f.z; // max refl
    if (++depth > 5)
        if (curand_uniform(state) < p)
            f = f * (1 / p);
        else
            return obj.e; // R.R.

    if (obj.refl == DIFF)
    { // Ideal DIFFUSE reflection
        double r1 = 2 * M_PI * curand_uniform(state), r2 = curand_uniform(state), r2s = sqrt(r2);
        float3 w = nl, u = normalize(modd((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w)), v = modd(w, u);
        float3 d = normalize((u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)));
        return obj.e + mult(f, radiance(Ray(x, d), depth, state));
    }
    else if (obj.refl == SPEC) // Ideal SPECULAR reflection
        return obj.e + mult(f, radiance(Ray(x, r.d - n * 2 * dot(n, r.d)), depth, state));
    Ray reflRay(x, r.d - n * 2 * dot(n, r.d)); // Ideal dielectric REFRACTION
    bool into = dot(n, nl) > 0;                // Ray from outside going in?
    double nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = dot(r.d, nl), cos2t;
    if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) < 0) // Total internal reflection
        return obj.e + mult(f, radiance(reflRay, depth, state));
    float3 tdir = normalize((r.d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))));
    double a = nt - nc, b = nt + nc, R0 = a * a / (b * b), c = 1 - (into ? -ddn : dot(tdir, n));
    double Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re, P = .25 + .5 * Re, RP = Re / P, TP = Tr / (1 - P);
    return obj.e + mult(f, depth > 2 ? (curand_uniform(state) < P ? // Russian roulette
                                            radiance(reflRay, depth, state) * RP
                                                                  : radiance(Ray(x, tdir), depth, state) * TP)
                                     : radiance(reflRay, depth, state) * Re + radiance(Ray(x, tdir), depth, state) * Tr);
}

__global__ void raytracer(float3 *image, int w, int h, int samps, ) int main(int argc, char *argv[])
{
    int w = 1024, h = 768, samps = argc == 2 ? atoi(argv[1]) / 4 : 1;              // # samples
    Ray cam(make_float3(50, 52, 295.6), normalize(make_float3(0, -0.042612, -1))); // cam pos, dir
    float3 cx = make_float3(w * .5135 / h, 0, 0), cy = normalize(modd(cx, cam.d)) * .5135, r, *c = new float3[w * h];
    // #pragma omp parallel for schedule(dynamic, 1) private(r) // OpenMP
    for (int y = 0; y < h; y++)
    { // Loop over image rows
        std::cout << "started\n";
        fprintf(stderr, "\rRendering (%d spp) %5.2f%%", samps * 4, 100. * y / (h - 1));
        for (unsigned short x = 0, Xi[3] = {0, 0, y * y * y}; x < w; x++) // Loop cols
            for (int sy = 0, i = (h - y - 1) * w + x; sy < 2; sy++)       // 2x2 subpixel rows
                for (int sx = 0; sx < 2; sx++, r = float3())
                { // 2x2 subpixel cols
                    for (int s = 0; s < samps; s++)
                    {
                        double r1 = 2 * curand_uniform(state), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
                        double r2 = 2 * curand_uniform(state), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
                        float3 d = cx * (((sx + .5 + dx) / 2 + x) / w - .5) +
                                   cy * (((sy + .5 + dy) / 2 + y) / h - .5) + cam.d;
                        d = normalize(d);
                        r = r + radiance(Ray(cam.o + d * 140, d), 0, state) * (1. / samps);
                    } // Camera rays are pushed ^^^^^ forward to start in interior
                    c[i] = c[i] + make_float3(clamp(r.x), clamp(r.y), clamp(r.z)) * .25;
                }
    }
    FILE *f = fopen("image.ppm", "w"); // Write image to PPM file.
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for (int i = 0; i < w * h; i++)
        fprintf(f, "%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));
}