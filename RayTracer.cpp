#include <iostream>
#include "parser.h"
#include "ppm.h"
#include <math.h>
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
struct color{
    int R;
    int G;
    int B;
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
double dotProduct(parser::Vec3f a, parser::Vec3f b)//?? neden cout basıyor? test için miyid? yoruma alıyorum, öyleyse sen silersin
{
    //cout << a.x << ", " << a.y << ", " << a.z << endl;
    //cout << b.x << ", " << b.y << ", " << b.z << endl;
    //cout << a.x*b.x << ", " << a.y*b.y << ", " << a.z*b.z << endl;
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
// length square function //?
double length2(parser::Vec3f a)
{
    return (a.x*a.x+a.y*a.y+a.z*a.z);
}
// length function
double length(parser::Vec3f a)
{
    return sqrt((a.x*a.x)+(a.y*a.y)+(a.z*a.z));
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
// compute clambed vector ---- should we check colors.x < 0 condition??
parser::Vec3i clamb(parser::Vec3f colors){
    parser::Vec3i clr;
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
parser::Vec3f add(parser::Vec3f a, parser::Vec3f b)
{
    parser::Vec3f tmp;
    tmp.x = a.x+b.x;
    tmp.y = a.y+b.y;
    tmp.z = a.z+b.z;
    return tmp;
}
//substract function
parser::Vec3f substract(parser::Vec3f a, parser::Vec3f b){

    parser::Vec3f result;
    result.x = b.x - a.x;
    result.y = b.y - a.y;
    result.z = b.z - a.z;

    return result;
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
//element-wise multiplication of two vectors Dot product 
parser::Vec3f elementMult(parser::Vec3f first, parser::Vec3f second){
    parser::Vec3f result;
    result.x = first.x * second.x;
    result.y = first.y * second.y;
    result.z = first.z * second.z;
    return result;
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
 
//finding normal for triangle
parser::Vec3f find_normal_t(parser::Scene scene, int k){

    parser::Vec3f normal;
    parser::Vec3f edge1;
    parser::Vec3f edge2;

    parser::Vec3f vertex0 = scene.vertex_data[scene.triangles[k].indices.v0_id -1];
    parser::Vec3f vertex1 = scene.vertex_data[scene.triangles[k].indices.v1_id -1];
    parser::Vec3f vertex2 = scene.vertex_data[scene.triangles[k].indices.v2_id -1];


    edge1 = substract(vertex0, vertex1);
    edge2 = substract(vertex0, vertex2);
    normal = crossProduct(edge1, edge2);

    return normal;
}
//finding normal for mesh
parser::Vec3f find_normal_m(parser::Scene scene, int k, int q){

    parser::Vec3f normal;
    parser::Vec3f edge1;
    parser::Vec3f edge2;

    parser::Vec3f vertex0 = scene.vertex_data[scene.meshes[k].faces[q].v0_id -1];
    parser::Vec3f vertex1 = scene.vertex_data[scene.meshes[k].faces[q].v1_id -1];
    parser::Vec3f vertex2 = scene.vertex_data[scene.meshes[k].faces[q].v2_id -1];


    edge1 = substract(vertex0, vertex1);
    edge2 = substract(vertex0, vertex2);
    normal = crossProduct(edge1, edge2);

    return normal;
}

// ray generation function 
Ray generateRay(int i, int j, parser::Camera cam)
{
    // ray representation -> r(t) = o + td
    // o-> origin, t->variable
    Ray tmp;
    // su = (i + 0.5)(r-l)/nx 
    // r->right, l->left, nx->image width
    // sv = (j + 0.5)(t-b)/ny
    // t->top, b->botton, ny->image height
    // m = e + -w(distance) --- w direction vector
    // q = m + lu + tv --- u&v are direction vectors
    // s = q + (su)u -(sv)v --- u&v are direction vectors
    //h: camera.nearplane gives us the coordinates o image as l,r,b,t. I think nearplane.x = l, .y= r, .z=b and .w = t
    //also camera.image width&heigh give the resolution. 
    //so, camera.imagewidth = # of cplumns, nx and 
    //camera.imageheight = # of rows, ny
    // for each camera different rays will be created
    // ...
    //alternative code according to my comments, we can rearrenge two code then.
    double su, sv;
    parser::Vec3f s, cam_u, q, m; //cam_u is the right dir. vector of camre. we compute u = v x w 
                               //q is the top-left corner coordinate vector of image
    double l = cam.near_plane.x;
    double r = cam.near_plane.y;
    double b = cam.near_plane.z;
    double t = cam.near_plane.w;
    int width = cam.image_width;
    int height = cam.image_height;

    su = ((i + 0.5) * ((r - l) / width));
    sv = ((j + 0.5) * ((t - b) / height));

    cam_u = crossProduct(cam.up, cam.gaze);
    m = add(cam.position, mult(cam.gaze, (-1)*cam.near_distance)); //m = q+ (-w + d) --> intersection point of image plane and gaze vector
    q = add(m, add(mult(cam_u, l), mult(cam.up, t)));
    s = add(q, add(mult(cam_u, su), mult(cam.up, (-1)*sv)));
    tmp.a = cam.position; 
    tmp.b = s; 
    return tmp;
}
// interseciton of sphere, returns t value
double intersectSphere(Ray r, parser::Sphere s, vector<parser::Vec3f> vertex)
{
    double A, B, C; // constants for the quadratic equation
    double delta; // solving for quadratic eqn.
    parser::Vec3f scenter;
    double sradius;
    parser::Vec3f p;
    double t, t1, t2;
    int i;
    scenter = vertex[s.center_vertex_id];
    sradius = s.radius;
    C = (r.a.x-scenter.x)*(r.a.x-scenter.x) + (r.a.y - scenter.y)*(r.a.y - scenter.y) + (r.a.z - scenter.z)*(r.a.z - scenter.z) - sradius*sradius;

    B = 2*r.b.x*(r.a.x - scenter.x) + 2*r.b.y*(r.a.y - scenter.y) + 2*r.b.z*(r.a.z - scenter.z);

    A = r.b.x*r.b.x + r.b.y*r.b.y + r.b.z*r.b.z;
    delta = (B*B) - (4*A*C);

    if(delta<0-EPSILON) { return -1;} // no solution for quadratic eqn.
    else if(delta <= 0+EPSILON && delta>=0-EPSILON) // has one distinct root
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
        if(t1>=1.0) t = t1;
        else t = -1;
    }
    return t;
}
// intersection of triangle, returns t value
double intersectTriangle(Ray r, parser::Triangle tri, vector<parser::Vec3f> vertex)
{
    // we'll use barycentric coordinate system
    // solve for t & beta & gamma 
    // by using matrices, determinant & Cramer rule
    //    |g a d|   |t    |   |j|
    //    |h b e| * |beta | = |k|
    //    |i c f|   |gamma|   |l|
    //    a -> v1, b -> v2, z -> v3
    double  a,b,c,d,e,f,g,h,i,j,k,l;
    double beta,gamma,t;
    
    double eimhf,gfmdi,dhmeg,akmjb,jcmal,blmkc;
    double M;
    
    double dd;
    parser::Vec3f ma,mb,mc;
    ma = vertex[tri.indices.v0_id];
    mb = vertex[tri.indices.v1_id];
    mc = vertex[tri.indices.v2_id];
    
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
    
    eimhf = e*i-h*f;
    gfmdi = g*f-d*i;
    dhmeg = d*h-e*g;
    akmjb = a*k-j*b;
    jcmal = j*c-a*l;
    blmkc = b*l-k*c;
    M = a*eimhf+b*gfmdi+c*dhmeg;
    if (M==0) return -1;
    
    t = -(f*akmjb+e*jcmal+d*blmkc)/M;
    
    if (t<1.0 + EPSILON) return -1;
    
    gamma = (i*akmjb+h*jcmal+g*blmkc)/M;
    
    if (gamma<0 || gamma>1) return -1;
    
    beta = (j*eimhf+k*gfmdi+l*dhmeg)/M;
    
    if (beta<0 || beta>(1-gamma)) return -1;
    
    return t;
}
//compute irradience E_i --> E_i = I / r^2 where r = |wi| and I is intensity
parser::Vec3f E_i(parser::Vec3f Intensity, parser::Vec3f w_i){
    parser::Vec3f result_E;
    float r = length(w_i);
    result_E = mult(Intensity, 1/(r*r));
    return result_E;
}
//compute ambient shading ---> L_a = k_a * I_a
parser::Vec3f L_a(int obj_id, parser::Scene scene){
    parser::Vec3f result_La;
    parser::Vec3f k_a = scene.materials[obj_id - 1].ambient; 
    parser::Vec3f I_a = scene.ambient_light;
    result_La = elementMult(k_a, I_a);
    return result_La;
}
//compute diffuse shading---> L_d = k_d * costheta * E_i and costheta = max(0, w_i * n)
parser::Vec3f L_d(parser::Scene scene, parser::Vec3f w_i, parser::Vec3f E_i, parser::Vec3f n, int obj_id){
    parser::Vec3f result_Ld;
    double cos_theta = max((double) 0, dotProduct(w_i, n));
    parser::Vec3f k_d = scene.materials[obj_id - 1].diffuse; 
    result_Ld = mult((elementMult(k_d, E_i)), cos_theta); 
    return result_Ld;
}
//compute specular shading --> L_s = k_s*(cosalpha)^p * E_i
parser::Vec3f L_s(parser::Scene scene, parser::Vec3f w_i, parser::Vec3f w_o, parser::Vec3f E_i, parser::Vec3f n, int obj_id){
    parser::Vec3f result_Ls;
    parser::Vec3f halfVector;
    parser::Vec3f k_s = scene.materials[obj_id - 1].diffuse;
    float cos_alpha;
    float phong = scene.materials[obj_id - 1].phong_exponent;
    halfVector = normalize(add(w_i, w_o));
    cos_alpha = max((double) 0, dotProduct(n, halfVector));
    result_Ls = mult(elementMult(k_s, E_i), (double) pow(cos_alpha, phong));
    return result_Ls;
}

bool shadow(parser::Scene scene, parser::Vec3f w_i, parser::Vec3f intersection, parser::Vec3f light_pos){

    const float Epsilon = scene.shadow_ray_epsilon;
    double t_s = 0.0f;
    parser::Vec3f reverse = substract(intersection, light_pos);
    double distance = length(reverse);
    parser::Vec3f new_origin = mult(w_i, Epsilon);

    Ray shadow_ray;
    shadow_ray.a = add(intersection, new_origin); //origin of shadow ray
    shadow_ray.b = w_i; //distance vector of shadow ray

    for(int i = 0; i < scene.triangles.size(); i++){

        double temp_t = intersectTriangle(shadow_ray, scene.triangles[i], scene.vertex_data);

        if(temp_t < 1.0 && temp_t > (0.0 + EPSILON)){

            return true;

        }
        
    }

    for(int i = 0; i < scene.spheres.size(); i++){

        double temp_t = intersectSphere(shadow_ray, scene.spheres[i], scene.vertex_data);

        if(temp_t < 1.0 && temp_t > (0.0 + EPSILON)){

            return true;
            
        }
        
    }

    return false;

}
void produce_image(parser::Scene scene, int width, int height, unsigned char* image)
{
    int i = 0;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            // for sphere
            // for mesh
            // for triangle
            // different loops for each other
            // hold intersecting point and if it is the first intersection point use light/shadow
            // consider multiple light sources
            Ray r; // ray
            double tmin = __DBL_MAX__;
            int closestTri, closestSphere, closestMesh, closestMeshTri;
            closestTri = -1;
            closestSphere = -1;
            closestMesh = -1;
            closestMeshTri = -1;
            r = generateRay(x,y, scene.cameras[0]);
            parser::Vec3f intersectionPoint;
            parser::Vec3f n; //normal vector
            parser::Vec3f w_i;
            parser::Vec3f I; 
            parser::Vec3f E;
            parser::Vec3f w_o;
            // check triangles
            for(int k = 0; k<scene.triangles.size(); k++)
            {
                double t;
                t = intersectTriangle(r, scene.triangles[k], scene.vertex_data);
                if(t>=1)
                {
                    if(t<tmin)
                    {
                        tmin = t;
                        closestTri = k; //closest_sphere = -1 ?? olması da gerekmiyor mu burada?
                        closestSphere = -1;
                        closestMesh = -1;
                        closestMeshTri = -1;
                        intersectionPoint = mult(r.b, t);
                        intersectionPoint = add(intersectionPoint, r.a);
                        n = find_normal_t(scene, k);

                    }
                }
            }

            // check meshes
            for(int k = 0; k<scene.meshes.size(); k++)
            {
                for(int q = 0; q<scene.meshes[k].faces.size(); q++)
                {    // each face is a triangle
                    double t;
                    parser::Triangle tri;
                    tri.indices.v0_id = scene.meshes[k].faces[q].v0_id;
                    tri.indices.v1_id = scene.meshes[k].faces[q].v1_id;
                    tri.indices.v2_id = scene.meshes[k].faces[q].v2_id;
                    tri.material_id = scene.meshes[k].material_id;
                    t = intersectTriangle(r, tri, scene.vertex_data);

                    if(t>=scene.cameras[0].near_distance-EPSILON && t<=scene.cameras[0].near_distance+EPSILON)
                    {
                        if(t<tmin)
                        {
                            tmin = t;
                            closestMesh = k;
                            closestMeshTri = q;
                            closestTri = -1;
                            intersectionPoint = mult(r.b, t);
                            intersectionPoint = add(intersectionPoint, r.a);
                            n = find_normal_m(scene, k, q);
                        }
                    }
                }
            }
            // check spheres
            for(int k = 0; k<scene.spheres.size(); k++)
            {
                double t;
                t = intersectSphere(r, scene.spheres[k], scene.vertex_data);
                if(t>=1)
                {
                    if(t<tmin)
                    {
                        tmin = t;
                        closestSphere = k;
                        closestMesh = -1;
                        closestMeshTri = -1;
                        closestTri = -1;

                        parser::Vec3f center = scene.vertex_data[scene.spheres[k].center_vertex_id - 1];

                        intersectionPoint = mult(r.b, t);
                        intersectionPoint = add(intersectionPoint, r.a);
                        n = substract(center, intersectionPoint);       
                    }
                }
            }    
            if(closestTri != -1) // the closest object that intersects with the ray is a triangle
            {
                // the closest object that intersects with the ray is a triangle
                // and we use scene.triangles[closestTri] to retrieve the data
                /*for(int l = 0; l<scene.point_lights.size(); l++)
                {
                    // compute the shadows ray s from x to I
                    //for each object p
                        // if s intersects p before the light source
                            // continue
                    // pixel color += L_d + L_s
                }*/
                // add L_a
                // for each light sources add L_d and L_s
                parser::Vec3f shading = L_a(scene.triangles[closestTri].material_id, scene); 
                image[i++] = shading.x;
                image[i++] = shading.y;
                image[i++] = shading.z;

                for(int l = 0; l < scene.point_lights.size(); l++){

                    w_i = substract(scene.point_lights[l].position, intersectionPoint);
                    w_i = normalize(w_i);
                    I = scene.point_lights[l].intensity; 
                    E = E_i(I, w_i);
                    w_o= normalize(substract(r.a,intersectionPoint));

                    if(shadow(scene, w_i, intersectionPoint, scene.point_lights[l].position)){ //if there is a shadow

                        continue;
                    }
                    else{

                        parser::Vec3f diffuse_s = L_d(scene, w_i, E, n, scene.triangles[closestTri].material_id);
                        parser::Vec3f specular_s = L_s(scene, w_i, w_o, E, n, scene.triangles[closestTri].material_id);
                        //L_m();

                        //shading = add(shading, diffuse_s);
                        //shading = add(shading, specular_s);
                        //shading = add(shading, mirror_s);
                        image[i-3] += diffuse_s.x + specular_s.x;
                        image[i-2] += diffuse_s.y + specular_s.y;
                        image[i-1] += diffuse_s.z + specular_s.z;
                    }
                }


            }
            else if(closestMeshTri != -1 && closestMesh != -1) 
            {
                // the closest object that intersects with the ray is a mesh
                // and we use scene.meshes[closestMesh].triangles[closestMeshTri] to retrieve the data
                // add L_a
                // for each light sources add L_d and L_s
                parser::Vec3f shading = L_a(scene.meshes[closestMesh].material_id , scene); 
                image[i++] = shading.x;
                image[i++] = shading.y;
                image[i++] = shading.z;
                for(int l = 0; l < scene.point_lights.size(); l++){

                    w_i = substract(scene.point_lights[l].position, intersectionPoint);
                    w_i = normalize(w_i);
                    I = scene.point_lights[l].intensity; 
                    E = E_i(I, w_i);
                    w_o= normalize(substract(r.a,intersectionPoint));

                    if(shadow(scene, w_i, intersectionPoint, scene.point_lights[l].position)){ //if there is a shadow

                        continue;
                    }
                    else{

                        parser::Vec3f diffuse_s = L_d(scene, w_i, E, n, scene.meshes[closestMesh].material_id);
                        parser::Vec3f specular_s = L_s(scene, w_i, w_o, E, n, scene.meshes[closestMesh].material_id);
                        //L_m();

                        //shading = add(shading, diffuse_s);
                        //shading = add(shading, specular_s);
                        //shading = add(shading, mirror_s);

                        image[i-3] = diffuse_s.x + specular_s.x;
                        image[i-2] = diffuse_s.y + specular_s.y;
                        image[i-1] = diffuse_s.z + specular_s.z;
                    }
                }
                

            }
            else if(closestSphere != -1) // the closest object that intersects with the ray is a sphere
            {
                // the closest object that intersects with the ray is a sphere
                // and we use scene.spheres[closestSphere] to retrieve the data
                // add L_a
                // for each light sources add L_d and L_s
                parser::Vec3f shading = L_a(scene.spheres[closestSphere].material_id, scene);
                image[i++] = shading.x;
                image[i++] = shading.y;
                image[i++] = shading.z;
                for(int l = 0; l < scene.point_lights.size(); l++){

                    w_i = substract(scene.point_lights[l].position, intersectionPoint);
                    w_i = normalize(w_i);
                    I = scene.point_lights[l].intensity; 
                    E = E_i(I, w_i);
                    w_o= normalize(substract(r.a,intersectionPoint));

                    if(shadow(scene, w_i, intersectionPoint, scene.point_lights[l].position)){ //if there is a shadow

                        continue;
                    }
                    else{

                        parser::Vec3f diffuse_s = L_d(scene, w_i, E, n, scene.triangles[closestSphere].material_id);
                        parser::Vec3f specular_s = L_s(scene, w_i, w_o, E, n, scene.triangles[closestSphere].material_id);
                        //L_m();

                        //shading = add(shading, diffuse_s);
                        //shading = add(shading, specular_s);
                        //shading = add(shading, mirror_s);
                        image[i-3] = diffuse_s.x + specular_s.x;
                        image[i-2] = diffuse_s.y + specular_s.y;
                        image[i-1] = diffuse_s.z + specular_s.z;
                        
                    }
                }

            }
            else
            {
                /* background color */
                parser::Vec3i shading = scene.background_color;
                image[i++] = shading.x;
                image[i++] = shading.y;
                image[i++] = shading.z;
            }

            /*int colIdx = x / columnWidth;
            image[i++] = BAR_COLOR[colIdx][0];
            image[i++] = BAR_COLOR[colIdx][1];
            image[i++] = BAR_COLOR[colIdx][2];*/
        }
    }
    write_ppm("test.ppm", image, width, height);
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
    produce_image(scene, width, height, image);
}
