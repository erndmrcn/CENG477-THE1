#include <iostream>
#include "parser.h"
#include "ppm.h"
#include <math.h>

#define ABS(a) ((a)>0?(a):-1*(a))
#define TRIANGLE 1
#define SPHERE 2
#define MESH 3
#define EPSILON 0.000000001

using namespace parser;
using namespace std;

typedef unsigned char RGB[3];

class Ray
{
    // represented like r(t) = e + dt 
public:
    // coordinates of the first vector (e)
    Vec3f a;
    // coordinates of the second vector (d)
    Vec3f b;
};

class Result
{
public:
    Vec3f intersectionPoint;
    Vec3f normal;
    double t;
    int material_id;
    int objType; 
    bool happened;
    Ray r;

};

struct color{
    int R;
    int G;
    int B;
};

Vec3f crossProduct(Vec3f a, Vec3f b)
{
    Vec3f tmp;
    tmp.x = a.y*b.z - a.z*b.y;
    tmp.y = a.z*b.x - a.x*b.z;
    tmp.z = a.x*b.y - a.y*b.x;
    return tmp;
}

double dotProduct(Vec3f a, Vec3f b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
// length square function //?
double length2(Vec3f a)
{
    return (a.x*a.x+a.y*a.y+a.z*a.z);
}
// length function
double length(Vec3f a)
{
    return sqrt((a.x*a.x)+(a.y*a.y)+(a.z*a.z));
}
// normalize function
Vec3f normalize(Vec3f v)
{
    Vec3f tmp;
    double l;
    l = length(v);
    tmp.x = v.x/l;
    tmp.y = v.y/l;
    tmp.z = v.z/l;
    return tmp;
}
// compute clambed vector ---- should we check colors.x < 0 condition??
Vec3i clamb(Vec3f colors){
    Vec3i clr;
    if(colors.x > 255){ clr.x = 255; }
    else if(colors.x < 0) { clr.x = 0; }
    else { clr.x = (int) round(colors.x); }
    if(colors.y > 255){ clr.y = 255; }
    else if(colors.y < 0) { clr.y = 0; }
    else { clr.y = (int) round(colors.y); }
    if(colors.z > 255){ clr.z = 255; }
    else if(colors.z < 0) { clr.z = 0; }
    else { clr.z = (int) round(colors.z); }
    return clr;
}
// add function
Vec3f add(Vec3f a, Vec3f b)
{
    Vec3f tmp;
    tmp.x = a.x+b.x;
    tmp.y = a.y+b.y;
    tmp.z = a.z+b.z;
    return tmp;
}
//substract function
Vec3f substract(Vec3f a, Vec3f b){
    Vec3f result;
    result.x = b.x - a.x;
    result.y = b.y - a.y;
    result.z = b.z - a.z;
    return result;
}
// multiplication of a matrix by a scalar
Vec3f mult(Vec3f v, double d)
{
    Vec3f tmp;
    tmp.x = v.x*d;
    tmp.y = v.y*d;
    tmp.z = v.z*d;
    return tmp;
}
//element-wise multiplication of two vectors Dot product 
Vec3f elementMult(Vec3f first, Vec3f second){
    Vec3f result;
    result.x = first.x * second.x;
    result.y = first.y * second.y;
    result.z = first.z * second.z;
    return result;
}
// distance between two vectors
double distance(Vec3f a, Vec3f b)
{
    parser::Vec3f t = substract(b,a);
    return sqrt(dotProduct(t,t));
}
// check if two vectors is equal
int equal(Vec3f a, Vec3f b)
{
    if((ABS((a.x-b.x))<EPSILON) && (ABS((a.y-b.y))<EPSILON) && (ABS((a.z-b.z))<EPSILON))
        return 1;
    else 
        return 0;
}
//finding normal for triangle
Vec3f find_normal_t(Scene scene, int k){

    Vec3f normal;
    Vec3f edge1;
    Vec3f edge2;

    Vec3f vertex0 = scene.vertex_data[scene.triangles[k].indices.v0_id -1];
    Vec3f vertex1 = scene.vertex_data[scene.triangles[k].indices.v1_id -1];
    Vec3f vertex2 = scene.vertex_data[scene.triangles[k].indices.v2_id -1];


    edge1 = substract(vertex0, vertex1);
    edge2 = substract(vertex0, vertex2);
    normal = crossProduct(edge1, edge2);
    normal = normalize(normal);
    return normal;
}
//finding normal for mesh
Vec3f find_normal_m(Scene scene, int k, int q){

    Vec3f normal;
    Vec3f edge1;
    Vec3f edge2;

    Vec3f vertex0 = scene.vertex_data[scene.meshes[k].faces[q].v0_id -1];
    Vec3f vertex1 = scene.vertex_data[scene.meshes[k].faces[q].v1_id -1];
    Vec3f vertex2 = scene.vertex_data[scene.meshes[k].faces[q].v2_id -1];


    edge1 = substract(vertex0, vertex1);
    edge2 = substract(vertex0, vertex2);
    normal = crossProduct(edge1, edge2);
    normal = normalize(normal);
    return normal;
}
// ray generation function 
Ray generateRay(int i, int j, Camera cam)
{
    // ray representation -> r(t) = o + td
    // o-> origin, t->variable
    Ray tmp;

    double su, sv;
    Vec3f s, cam_u, q, m; //cam_u is the right dir. vector of camre. we compute u = v x w 
                               //q is the top-left corner coordinate vector of image
    double l = cam.near_plane.x;
    double r = cam.near_plane.y;
    double b = cam.near_plane.z;
    double t = cam.near_plane.w;
    int width = cam.image_width;
    int height = cam.image_height;
    
    su = ((i + 0.5) * ((r - l) / width));
    sv = ((j + 0.5) * ((t - b) / height));
    Vec3f w = mult(cam.gaze,-1);
    cam_u = crossProduct(cam.up, w);

    m = add(cam.position, mult(cam.gaze, cam.near_distance)); //m = q+ (-w*d) --> intersection point of image plane and gaze vector
    q = add(m, add(mult(cam_u, l), mult(cam.up, t)));
    s = add(q, add(mult(cam_u, su), mult(cam.up, (-1)*sv)));
    tmp.a = cam.position; 
    tmp.b = substract(cam.position,s); 
    return tmp;
}
// interseciton of sphere, returns t value
double intersectSphere(Ray r, Sphere s, vector<Vec3f> vertex)
{
    double A, B, C; // constants for the quadratic equation
    double delta; // solving for quadratic eqn.
    Vec3f scenter;
    double sradius;
    Vec3f p;
    double t, t1, t2;
    int i;
    scenter = vertex[s.center_vertex_id-1];
    sradius = s.radius;
    C = (r.a.x-scenter.x)*(r.a.x-scenter.x) + (r.a.y - scenter.y)*(r.a.y - scenter.y) + (r.a.z - scenter.z)*(r.a.z - scenter.z) - sradius*sradius;

    B = 2*r.b.x*(r.a.x - scenter.x) + 2*r.b.y*(r.a.y - scenter.y) + 2*r.b.z*(r.a.z - scenter.z);

    A = r.b.x*r.b.x + r.b.y*r.b.y + r.b.z*r.b.z;
    delta = (B*B) - (4*A*C);
    if(delta<0) { return -1;} // no solution for quadratic eqn.
    else if(delta <= 0 && delta>=0) // has one distinct root
    {   
        t = -B / (2*A);
    } 
    else
    {
        double tmp;
        delta = sqrt(delta);
        A = 2*A;
        t1 = (-B + delta) / A;
        t2 = (-B - delta) / A;
        if(t2<t1){
            tmp = t2;
            t2 = t1;
            t1 = tmp;
        }
        if(t1>=0.0) t = t1;
        else t = -1;
    }
    return t;
}

// intersection of triangle, returns t value
double intersectTriangle(Ray r, Triangle tri, vector<Vec3f> vertex)
{

    double  a,b,c,d,e,f,g,h,i,j,k,l;
    double beta,gamma,t;
    
    double M;
    
    double dd;
    Vec3f ma,mb,mc;
    ma = vertex[tri.indices.v0_id-1];
    mb = vertex[tri.indices.v1_id-1];
    mc = vertex[tri.indices.v2_id-1];
    
    a = ma.x-mb.x;
    b = ma.y-mb.y;
    c = ma.z-mb.z;
    d = ma.x-mc.x;
    e = ma.y-mc.y;
    f = ma.z-mc.z;
    
    g = r.b.x;
    h = r.b.y;
    i = r.b.z;
    
    j = ma.x-r.a.x;
    k = ma.y-r.a.y;
    l = ma.z-r.a.z;

    M = (a*(e*i - f*h)) -
        (d*(b*i - h*c)) +
        (g*(b*f - e*c));

    if (M==0.0) return -1;

    t = ((a*(e*l - f*k)) -
        (d*(b*l - k*c) )+
       (j*(b*f - e*c)))/(M);

    if (t<=0.0) return -1;
    
    gamma = ((a*(k*i - l*h)) - (j*(b*i -c*h)) + (g*(b*l - c*k)) )/M;
    
    if (gamma<0 || gamma>1) return -1;
    
    beta = ((j*(e*i - f*h)) - (d*(k*i -l*h)) + (g*(k*f - e*l)) )/M;
    
    if (beta<0 || beta>(1-gamma)) return -1;
    
    return t;
}
//compute irradience E_i --> E_i = I / r^2 where r = |wi| and I is intensity
Vec3f E_i(Vec3f Intensity, Vec3f w_i, double r){
    Vec3f result_E;
    result_E = mult(Intensity, 1/(r*r));
    return result_E;
}
//compute ambient shading ---> L_a = k_a * I_a
Vec3f L_a(int obj_id, Scene scene){
    Vec3f result_La;
    Vec3f k_a = scene.materials[obj_id-1].ambient; 
    Vec3f I_a = scene.ambient_light;
    result_La = elementMult(k_a, I_a);
    return result_La;
}
//compute diffuse shading---> L_d = k_d * costheta * E_i and costheta = max(0, w_i * n)
Vec3f L_d(Scene scene, Vec3f w_i, Vec3f E_i, Vec3f n, int obj_id){
    Vec3f result_Ld;
    double cos_theta = max((double) 0, dotProduct(w_i, n));
    Vec3f k_d = scene.materials[obj_id-1].diffuse; 
    result_Ld = mult((elementMult(k_d, E_i)), cos_theta); 
    return result_Ld;
}
//compute specular shading --> L_s = k_s*(cosalpha)^p * E_i
Vec3f L_s(Scene scene, Vec3f w_i, Vec3f w_o, Vec3f E_i, Vec3f n, int obj_id){
    Vec3f result_Ls;
    Vec3f halfVector;
    Vec3f k_s = scene.materials[obj_id-1].specular;
    float cos_alpha;
    float phong = scene.materials[obj_id-1].phong_exponent;
    halfVector = normalize(add(w_i, w_o));
    cos_alpha = max((double) 0, dotProduct(n, halfVector));
    result_Ls = mult(elementMult(k_s, E_i), (double) pow(cos_alpha, phong));
    return result_Ls;
}

bool isMirror(Scene scene, int material_id)
{
    if(scene.materials[material_id-1].mirror.x>0 || scene.materials[material_id-1].mirror.y>0 || scene.materials[material_id-1].mirror.z>0)
    {
        return true;
    }
    return false;
}
Result trace(Scene scene, Ray r, Camera cam)
{
	Vec3f shading;
    Result res;
    res.happened = false;
	double tmin = __DBL_MAX__;    
    Vec3f intersectionPoint;
    Vec3f n; //normal vector
    
    //check triangles
    int triangleSize = scene.triangles.size();
    for(int k = 0; k<triangleSize; k++)
    {
        double t;
        t = intersectTriangle(r, scene.triangles[k], scene.vertex_data);
        if(t>=0.0)
        {
            if(t<tmin)
            {
                tmin = t;
                res.intersectionPoint = mult(r.b, t);
                res.intersectionPoint = add(intersectionPoint, r.a);
                res.normal = find_normal_t(scene, k);
                res.material_id = scene.triangles[k].material_id;
                res.objType = TRIANGLE;
                res.t = t;
                res.happened = 1;
                res.r = r;
                }
            }
    }
    // check meshes
    int meshSize = scene.meshes.size();
    for(int k = 0; k<meshSize; k++)
    {
        int faceSize = scene.meshes[k].faces.size();
        for(int q = 0; q<faceSize; q++)
        {    // each face is a triangle
                double t;
                Triangle tri;
                tri.indices.v0_id = scene.meshes[k].faces[q].v0_id;
                tri.indices.v1_id = scene.meshes[k].faces[q].v1_id;
                tri.indices.v2_id = scene.meshes[k].faces[q].v2_id;
                tri.material_id = scene.meshes[k].material_id;
                t = intersectTriangle(r, tri, scene.vertex_data);

                if(t>=0.0)
                {
                    if(t<tmin)
                    {
                        tmin = t;
                        res.intersectionPoint = mult(r.b, t);
                        res.intersectionPoint = add(res.intersectionPoint, r.a);
                        res.normal = find_normal_m(scene, k, q);
                        res.normal = normalize(res.normal);
                        res.material_id = scene.meshes[k].material_id;
                        res.objType = MESH;
                        res.t = t;
                        res.happened = 1;
                        res.r = r;
                    }
                }
            }
        }
        // check spheres
        int sphereSize = scene.spheres.size();
        for(int k = 0; k<sphereSize; k++)
        {
            double t;
            t = intersectSphere(r, scene.spheres[k], scene.vertex_data);
            if(t>=0.0)
            {
                if(t<tmin)
                {
                    tmin = t;
                    res.happened = 1;
                    res.t = t;
                    res.material_id = scene.spheres[k].material_id;
                    res.objType = SPHERE;
                    Vec3f center = scene.vertex_data[scene.spheres[k].center_vertex_id - 1];
                    double radius = scene.spheres[k].radius;
                    res.intersectionPoint = mult(r.b, t);
                    res.intersectionPoint = add(res.intersectionPoint, r.a);
                    res.normal = substract(center, res.intersectionPoint);
                    res.normal.x /= radius;
                    res.normal.y /= radius;
                    res.normal.z /= radius;
                    res.normal = normalize(res.normal); 
                    res.r = r;      
                }
            }
        }

        return res;
}
Vec3f shadow(Scene scene, Result res, Camera cam, int depth)
{
    const float shadow_epsilon = scene.shadow_ray_epsilon;
    Vec3f w_i, w_o, irradiance;
    Ray shadow_ray;
    bool shadow_flag = false;
    double tLight, temp_t, distance1;

    float pixel1 = 0;
    float pixel2 = 0;
    float pixel3 = 0;

    if(res.happened == true)
    {
        int material_id = res.material_id;
        Vec3f ambient = L_a(material_id, scene);
        pixel1 += ambient.x;
        pixel2 += ambient.y;
        pixel3 += ambient.z;
        int lightSize = scene.point_lights.size();
        for(int l = 0; l<lightSize; l++)
        {
            PointLight currentLight = scene.point_lights[l];
            w_i = substract(res.intersectionPoint,currentLight.position);
            w_o = substract(res.intersectionPoint, cam.position);
            distance1 = length(w_i);
            irradiance = E_i(currentLight.intensity, w_i, length(w_i));
            shadow_ray.a = add(res.intersectionPoint, mult(w_i, shadow_epsilon));
            shadow_ray.b = w_i;
            tLight = substract(shadow_ray.a, currentLight.position).x / shadow_ray.b.x;
            double lightToCam = distance(currentLight.position, cam.position);

            int sphereSize = scene.spheres.size();
            for(int sp = 0; sp<sphereSize; sp++)
            {
                temp_t = intersectSphere(shadow_ray, scene.spheres[sp], scene.vertex_data);
                if(tLight>=temp_t && temp_t>0)
                {
                    shadow_flag = true;
                }
            }
            int triangleSize = scene.triangles.size();
            for(int tr = 0; tr<triangleSize; tr++)
            {   
                temp_t = intersectTriangle(shadow_ray, scene.triangles[tr], scene.vertex_data);
                if(temp_t != -1)
                if(temp_t<=tLight && temp_t>0)
                {   
                    shadow_flag = true;
                }
            }
            if(!shadow_flag)
            {
                int meshSize = scene.meshes.size();
                for(int mh = 0; mh<meshSize; mh++)
                {
                    int faceSize = scene.meshes[mh].faces.size();
                    for(int f = 0; f<faceSize; f++)
                    {
                        Face face = scene.meshes[mh].faces[f];
                        Triangle tri;
                        tri.indices.v0_id = face.v0_id;
                        tri.indices.v1_id = face.v1_id;
                        tri.indices.v2_id = face.v2_id;
                        tri.material_id = scene.meshes[mh].material_id;
                        temp_t = intersectTriangle(shadow_ray, tri, scene.vertex_data);
                        if(tLight>=temp_t && temp_t>0)
                        {
                            shadow_flag = true;
                        }
                    }
                }
            }
            if(!shadow_flag || (shadow_flag&&lightToCam == 0.999)){
                if (res.objType == TRIANGLE) cout << "tri" << endl;
                Vec3f diffuse = L_d(scene, w_i, irradiance, res.normal, material_id);
                Vec3f specular = L_s(scene, w_i, w_o, irradiance, res.normal, material_id);

                pixel1 += diffuse.x + specular.x;
                pixel2 += diffuse.y + specular.y;
                pixel3 += diffuse.z + specular.z;
            }
        }
        bool reflect = isMirror(scene, material_id);
        Vec3f reflection;
        reflection.x = 0;
        reflection.y = 0;
        reflection.z = 0;

        if(depth > 0 && reflect)
        {
            double w_i = -2*dotProduct(res.r.b,res.normal);
            Vec3f normal_w_i;
            normal_w_i.x = res.normal.x*w_i + res.r.b.x;
            normal_w_i.y = res.normal.y*w_i + res.r.b.y;
            normal_w_i.z = res.normal.z*w_i + res.r.b.z;

            normal_w_i = normalize(normal_w_i);

            Vec3f wi_epsilon;
            wi_epsilon.x = normal_w_i.x*shadow_epsilon;
            wi_epsilon.y = normal_w_i.y*shadow_epsilon;
            wi_epsilon.z = normal_w_i.z*shadow_epsilon;

            Ray reflectionRay;
            reflectionRay.a = add(res.intersectionPoint, wi_epsilon);
            reflectionRay.b = normal_w_i;

            Result reflectResult = trace(scene, reflectionRay, cam);
            if(!(reflectResult.material_id == res.material_id && reflectResult.objType == res.objType))
            {
                reflection = shadow(scene, reflectResult, cam, depth-1);
            }

            pixel1 += reflection.x * scene.materials[material_id-1].mirror.x;
            pixel2 += reflection.y * scene.materials[material_id-1].mirror.y;
            pixel3 += reflection.z * scene.materials[material_id-1].mirror.z;
        }
    }
    else
    {
        /* Backgroung color */
        pixel1 = scene.background_color.x;
        pixel2 = scene.background_color.y;
        pixel3 = scene.background_color.z;
    }

    Vec3f pixelColor;
    pixelColor.x = pixel1;
    pixelColor.y = pixel2;
    pixelColor.z = pixel3;
    return pixelColor;
}

int main(int argc, char* argv[])
{
    // Sample usage for reading an XML scene file
    Scene scene;
    scene.loadFromXml(argv[1]);

    int pixel = 0;
    int cameraSize = scene.cameras.size();
    for(int i = 0; i < cameraSize; ++i)
    {
        Camera currentCamera = scene.cameras[i];
    	int width = scene.cameras[i].image_width;
        int height = scene.cameras[i].image_height;
    	unsigned char* image = new unsigned char [width * height * 3];
        for(int x = 0; x<height; x++)
        {
            for(int y = 0; y<width; y++)
            {
                Ray r = generateRay(y, x, currentCamera);

                Result res = trace(scene, r, currentCamera);

                Vec3f colorf = shadow(scene, res, currentCamera, scene.max_recursion_depth);

                Vec3i color = clamb(colorf);
                image[pixel++] = color.x;
                image[pixel++] = color.y;
                image[pixel++] = color.z;
            }
        }
        write_ppm(currentCamera.image_name.c_str(), image, width, height);	
    }
    return 0;
}
