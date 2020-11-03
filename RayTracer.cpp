#include <iostream>
#include "parser.h"
#include "ppm.h"

#define ABS(a) ((a)>0?(a):-1*(a))
#define EPSILON 0.000000001

typedef unsigned char RGB[3];
using namespace std;

class Ray
{
    // represented like r(t) = e + dt 
public:
    // coordinates of the first vector (e)
    parser::Vec3f a;
    // coordinates of the second vector (d)
    parser::Vec3f b;
};

// Cross product of 2 vectors
// i*i = 0(i), i*j = 1(k), i*k = 1(j)
// j*j = 0(j), j*i = -1(k), j*k = 1(i)
// k*k = 0(k), k*i = -1(j), k*j = -1(i)
// x -> i*i + j*k + k*j
// y -> j*j + i*k + k*i
// z -> k*k + i*j + j*i
parser::Vec3f crossProduct(parser::Vec3f a, parser::Vec3f b)
{
    parser::Vec3f tmp;

    tmp.x = a.y*b.z - a.z*b.y;
    tmp.y = a.z*b.x - a.x*b.z;
    tmp.z = a.x*b.y - a.y*b.x;

    return tmp;
}

// Dot product of 2 vectors
// i*i = 1, i*j = 0, i*k = 0
double dotProduct(parser::Vec3f a, parser::Vec3f b)
{
    return a.x*b.x+a.y*b.y+a.z*b.z;
}

// length square function
double length2(parser::Vec3 a)
{
    return (a.x*a.x+a.y*a.y+a.z*a.z);
}

// length function
double length(parser::Vec3 a)
{
    return sqrt(a.x*a.x+a.y*a.y*+a.z*a.z)
}

// normalize function
parser::Vec3f normalize(parser::Vec3f v)
{
    parser::Vec3f tmp;
    double l;

    l = length(v);
    tmp.x = v.x/l;
    tmp.y = v.y/l;
    tmp.z = v.z/l;

    return tmp;
}

// add function
parser::Vec3f add(parser::Vec3 a, parser::Vec3 b)
{
    parser::Vec3f tmp;
    tmp.x = a.x+b.x;
    tmp.y = a.y+b.y;
    tmp.z = a.z+b.z;

    return tmp;
}

// multiplication of a matrix by a scalar
parser::Vec3f mult(parser::Vec3f v, double d)
{
    parser::Vec3f tmp;
    tmp.x = v.x*d;
    tmp.y = v.y*d;
    tmp.z = v.z*d;

    return tmp;
}
// distance between two vectors
double distance(parser::Vec3f a, parser::Vec3f b)
{
    return sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z));
}

// check if two vectors is equal
int equal(parser::Vec3f a, parser::Vec3f b)
{
    if((ABS((a.x-b.x))<EPSILON) && (ABS((a.y-b.y))<EPSILON) && (ABS((a.z-b.z))<EPSILON))
        return 1;
    else 
        return 0;
}

// ray generation function 
Ray generateRay(int i, int j, parser::Camera cam)
{
    // ray representation -> r(t) = o + td
    // o-> origin, t->variable
    Ray tmp;
    // su = (i + 0.5)(r-l)/nx 
    // r->right, l->left, nx->image width
    // sv = (i + 0.5)(t-b)/ny
    // t->top, b->botton, ny->image height
    // m = e + -w(distance) --- w direction vector
    // q = m + lu + tv --- u&v are direction vectors
    // s = q + (su)u -(sv)v --- u&v are direction vectors

    parser::Vec3f su, sv, s;
    double pixelW = (cam.-cam.l)/(double)sizeX;
    double halfPixelW = pixelW*0.5;
    double pixelH = ()
    tmp.a = cam.position;
    su = mult(cam.up,cam.l+(i*cam.image_width)+)

    // for each camera different rays will be created
    // ...
}

// interseciton of sphere, returns t value
double intersectSphere(Ray r, parser::Sphere s)
{
    double A, B, C; // constants for the quadratic equation

    double delta; // solving for quadratic eqn.
    parser::Vec3f scenter;
    double sradius;
    parser::Vec3f p;
    double t, t1, t2;
    int i;

    scenter = s.center;
    sradius = s.r;

    // calculate A B C and solve for t
    // return t
}

// intersection of triangle, returns t value
double intersectTriangle(Ray r, parser::Triangle tri)
{
    // we'll use barycentric coordinate system
}
int main(int argc, char* argv[])
{
    // Sample usage for reading an XML scene file
    parser::Scene scene;
    scene.loadFromXml(argv[1]);
    /****
     **** 
            mesh -> scene.meshes -> vector
            faces -> scene.meshes[i].faces -> vector
            vertex_id -> scene.meshes[i].faces[j].v0_id -> vertex0 of ith mesh jth faces -> int
            vertex_data -> scene.vertex_data[scene.meshes[i].faces[j].v0_id].x
                                                                            .y
                                                                            .z -> we can reach each vertex coordinate
     ****
    ****/

    // The code below creates a test pattern and writes
    // it to a PPM file to demonstrate the usage of the
    // ppm_write function.
    //
    // Normally, you would be running your ray tracing
    // code here to produce the desired image.

    const RGB BAR_COLOR[8] =
    {
        { 255, 255, 255 },  // 100% White
        { 255, 255,   0 },  // Yellow
        {   0, 255, 255 },  // Cyan
        {   0, 255,   0 },  // Green
        { 255,   0, 255 },  // Magenta
        { 255,   0,   0 },  // Red
        {   0,   0, 255 },  // Blue
        {   0,   0,   0 },  // Black
    };

    int width = 640, height = 480;
    int columnWidth = width / 8;

    unsigned char* image = new unsigned char [width * height * 3];

    int i = 0;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int colIdx = x / columnWidth;
            image[i++] = BAR_COLOR[colIdx][0];
            image[i++] = BAR_COLOR[colIdx][1];
            image[i++] = BAR_COLOR[colIdx][2];
        }
    }

    write_ppm("test.ppm", image, width, height);

}
