#include <iostream>
#include "parser.h"
#include "ppm.h"
#include <math.h>
#define ABS(a) ((a)>0?(a):-1*(a))
#define TMAX 200000.0
typedef unsigned char RGB[3];
using namespace std;
using namespace parser;
constexpr float kEpsilon = 1e-8; 
class Ray
{
    // represented like r(t) = e + dt 
public:
    // coordinates of the first vector (e)
    Vec3f a;
    // coordinates of the second vector (d)
    Vec3f b;
};
struct color{
    int R;
    int G;
    int B;
};

Vec3f trace(Scene scene, Camera cam, Ray r, int depth);

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
Vec3f clamb(Vec3f colors){
    Vec3f clr;
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
    return sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z));
}
// check if two vectors is equal
int equal(Vec3f a, Vec3f b)
{
    if((ABS((a.x-b.x))<kEpsilon) && (ABS((a.y-b.y))<kEpsilon) && (ABS((a.z-b.z))<kEpsilon))
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
double intersectSphere(Ray r, Sphere s, Vec3f scenter, double sradius)
{
        double A, B, C; // constants for the quadratic equation
    double delta; // solving for quadratic eqn.
    Vec3f p;
    double t, t1, t2;
    int i;
    p = substract(scenter, r.a);
    C = dotProduct(p,p)-sradius*sradius;

    B = 2*dotProduct(p,r.b);

    A = dotProduct(r.b, r.b);
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
double intersectTriangle(Ray r, Triangle tri, Vec3f ma, Vec3f mb, Vec3f mc)
{

    /*double  a,b,c,d,e,f,g,h,i,j,k,l;
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
    if (M==0) return -1;
    t = ((a*(e*l - f*k)) -
        (d*(b*l - k*c) )+
       (j*(b*f - e*c)))/(M);
    if (t<1.0) return -1;
    
    gamma = ((a*(k*i - l*h)) - (j*(b*i -c*h)) + (g*(b*l - c*k)) )/M;
    
    if (gamma<0 || gamma>1) return -1;
    
    beta = ((j*(e*i - f*h)) - (d*(k*i -l*h)) + (g*(k*f - e*l)) )/M;
    
    if (beta<0 || beta>(1-gamma)) return -1;
    
    return t;*/

    Vec3f  edge1, edge2;
    double u,v,t;

    edge1 = substract(ma, mb);
    edge2 = substract(ma, mc);

    Vec3f pvec = crossProduct(r.b, edge2);
    float det = dotProduct(edge1, pvec);

    if( det < kEpsilon){

    	t = -1;
    	return t;

    } 
    if(fabs(det) < kEpsilon){
    	t = -1;
    	return t;	
    } 

    float invDet = 1 / det;

    Vec3f tvec = substract(ma, r.a);
	u = dotProduct(tvec, pvec) * invDet;

    if(u < 0 || u > 1){
    	t = -1;
    	return t;

    } 

    Vec3f qvec = crossProduct(tvec, edge1);
	v = dotProduct(r.b, qvec) * invDet;

    if(v < 0 || u+v > 1){
    	t = -1;
    	return t;
    } 

    t = dotProduct(edge2, qvec) * invDet;
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

Vec3f L_m(Camera cam, Scene scene, Vec3f w_o, Vec3f intersection, Vec3f n, int obj_id, int depth){

	Vec3f result_Lm;

	Vec3f k_m = scene.materials[obj_id -1].mirror;

	if(k_m.x == 0 && k_m.y == 0 && k_m.z == 0){

		result_Lm.x = 0;
		result_Lm.y = 0;
		result_Lm.z = 0;

	}
	else{

		
		Vec3f w_o2;
		Vec3f w_r;
		Ray rr;

		w_o2 = mult(n, 2*dotProduct(w_o, n));

		w_r = add(mult(w_o, -1), w_o2);
		w_r = normalize(w_r);

		rr.a.x = intersection.x; //+ scene.shadow_ray_epsilon;
		rr.a.y = intersection.y; //+ scene.shadow_ray_epsilon;
		rr.a.z = intersection.z; //+ scene.shadow_ray_epsilon;

		rr.b = w_r;

		result_Lm = trace(scene, cam, rr, depth);

		result_Lm = elementMult(result_Lm, k_m);

		
	}

	return result_Lm;
}

bool shadow(Scene scene, Vec3f w_i, Vec3f intersection, Vec3f light_pos){
    const float Epsilon = scene.shadow_ray_epsilon;
    double t_s = 0.0f;
    Vec3f reverse = substract(intersection, light_pos);
    double distance = length(reverse);
    Vec3f new_origin = mult(w_i, Epsilon);

    Ray shadow_ray;
    shadow_ray.a = add(intersection, new_origin); //origin of shadow ray
    shadow_ray.b = w_i; //distance vector of shadow ray

    for(int i = 0; i < scene.triangles.size(); i++){
        Triangle triangle = scene.triangles[i];
        Vec3f v1, v2, v3;
        v1 = scene.vertex_data[triangle.indices.v0_id-1];
        v2 = scene.vertex_data[triangle.indices.v1_id-1];
        v3 = scene.vertex_data[triangle.indices.v2_id-1];
        double temp_t = intersectTriangle(shadow_ray, triangle, v1, v2, v3);

        if(temp_t > (0.0-Epsilon) && temp_t < distance){
            return true;
        }
        
    }
    for(int i = 0; i<scene.meshes.size(); i++)
    {
        for(int j = 0; j<scene.meshes[i].faces.size(); j++)
        {
            Face face = scene.meshes[i].faces[j];
            Triangle tri1;
            tri1.indices.v0_id = face.v0_id;
            tri1.indices.v1_id = face.v1_id;
            tri1.indices.v2_id = face.v2_id;
            tri1.material_id = scene.meshes[i].material_id;
            Vec3f v1, v2, v3;
            v1 = scene.vertex_data[tri1.indices.v0_id-1];
            v2 = scene.vertex_data[tri1.indices.v1_id-1];
            v3 = scene.vertex_data[tri1.indices.v2_id-1];
        
            double temp_t = intersectTriangle(shadow_ray, tri1, v1, v2, v3);

            if(temp_t> (0.0 - Epsilon) && temp_t < distance){
                
                 return true;
            }
        }
    }
    for(int i = 0; i < scene.spheres.size(); i++){
        Sphere sphere = scene.spheres[i];
        Vec3f center = scene.vertex_data[sphere.center_vertex_id-1];
        double temp_t = intersectSphere(shadow_ray, sphere, center, sphere.radius);
        
        if((temp_t > 0.0) && temp_t < distance){
            
            return true;
        }
    }
    return false;
}

Vec3f trace(Scene scene, Camera camera, Ray r, int depth){

	Vec3f shading;

	if(depth == 0){

		
		shading.x = 0;
		shading.y = 0;
		shading.z = 0;
	}
	else{

		depth--;

		double tmin = __DBL_MAX__;
        
        int closestTri, closestSphere, closestMesh, closestMeshTri;
        Vec3f intersectionPoint;
        Vec3f n; //normal vector
        Vec3f w_i;
        Vec3f I; 
        Vec3f E;
        Vec3f w_o;
        

        closestTri = -1;
        closestSphere = -1;
        closestMesh = -1;
        closestMeshTri = -1;

        //check triangles
        for(int k = 0; k<scene.triangles.size(); k++){
                double t;
                Triangle triangle = scene.triangles[k];
                Vec3f v1, v2, v3;
                v1 = scene.vertex_data[triangle.indices.v0_id-1];
                v2 = scene.vertex_data[triangle.indices.v1_id-1];
                v3 = scene.vertex_data[triangle.indices.v2_id-1];
                t = intersectTriangle(r, triangle, v1, v2, v3);
               if(t>=0)
                {
                    if(t<tmin)
                    {
                        tmin = t;
                        closestTri = k; 
                        closestSphere = -1;
                        closestMesh = -1;
                        closestMeshTri = -1;

                        intersectionPoint = mult(r.b, t);
                        intersectionPoint = add(intersectionPoint, r.a);

                        n = find_normal_t(scene, k);
                        n = normalize(n);
                    }
                }
        }
            // check meshes
            for(int k = 0; k<scene.meshes.size(); k++)
            {
                for(int q = 0; q<scene.meshes[k].faces.size(); q++)
                {    // each face is a triangle
                    double t;
                    Triangle tri;
                    tri.indices.v0_id = scene.meshes[k].faces[q].v0_id;
                    tri.indices.v1_id = scene.meshes[k].faces[q].v1_id;
                    tri.indices.v2_id = scene.meshes[k].faces[q].v2_id;
                    tri.material_id = scene.meshes[k].material_id;
                    Vec3f v1, v2, v3;
                    v1 = scene.vertex_data[tri.indices.v0_id-1];
                    v2 = scene.vertex_data[tri.indices.v1_id-1];
                    v3 = scene.vertex_data[tri.indices.v2_id-1];
                    t = intersectTriangle(r, tri, v1, v2, v3);

                    if(t>=0)
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
                            n = normalize(n);
                        }
                    }
                }
            }
            // check spheres
            for(int k = 0; k<scene.spheres.size(); k++)
            {
                double t;
                Sphere sphere = scene.spheres[k];
                Vec3f center = scene.vertex_data[sphere.center_vertex_id-1];
                t = intersectSphere(r, sphere, center, sphere.radius);
                if(t>=0)
                {
                    if(t<tmin)
                    {
                        tmin = t;
                        closestSphere = k;
                        closestMesh = -1;
                        closestMeshTri = -1;
                        closestTri = -1;

                        Vec3f center = scene.vertex_data[scene.spheres[k].center_vertex_id - 1];

                        intersectionPoint = mult(r.b, t);
                        intersectionPoint = add(intersectionPoint, r.a);
                        n = substract(center, intersectionPoint);
                        n.x /= scene.spheres[k].radius;
                        n.y /= scene.spheres[k].radius;
                        n.z /= scene.spheres[k].radius;
                        n = normalize(n);       
                    }
                }
            }
            if(closestTri != -1) // the closest object that intersects with the ray is a triangle
            {
                shading = L_a(scene.triangles[closestTri].material_id, scene);
                

                for(int l = 0; l < scene.point_lights.size(); l++){
                    w_i = substract(intersectionPoint,scene.point_lights[l].position);
                    double distance = length(w_i);
                    I = scene.point_lights[l].intensity; 
                    E = E_i(I, w_i,distance);
                    w_o = substract(intersectionPoint, r.a);
                    w_i = normalize(w_i);
                    w_o = normalize(w_o);
                    if(shadow(scene, w_i, intersectionPoint, scene.point_lights[l].position)){ //if there is a shadow
                        continue;
                    }
                    else{
                        Vec3f diffuse_s = L_d(scene, w_i, E, n, scene.triangles[closestTri].material_id);
                        Vec3f specular_s = L_s(scene, w_i, w_o, E, n, scene.triangles[closestTri].material_id);
                        //shading = add(shading, L_m(camera, scene, r, intersectionPoint, n, scene.triangles[closestTri].material_id, depth));
                        shading = add(add(diffuse_s,shading),specular_s); 
                        
                        
                    }
                    
                }

                shading = add(shading, L_m(camera, scene, w_o, intersectionPoint, n, scene.triangles[closestTri].material_id, depth));
            }
            else if(closestMeshTri != -1 && closestMesh != -1) 
            {

                shading = L_a(scene.meshes[closestMesh].material_id , scene); 
                 
               	for(int l = 0; l < scene.point_lights.size(); l++){
	                w_i = substract(intersectionPoint,scene.point_lights[l].position);
	                double distance = length(w_i);
	                I = scene.point_lights[l].intensity; 
	                E = E_i(I, w_i,distance);
	                w_o = substract(intersectionPoint,r.a);
	                w_i = normalize(w_i);
	                w_o = normalize(w_o);
	                    
	                if(shadow(scene, w_i, intersectionPoint, scene.point_lights[l].position)){ //if there is a shadow
	                    continue;
	                }
	                else{

	                    Vec3f diffuse_s = L_d(scene, w_i, E, n, scene.meshes[closestMesh].material_id);
	                    Vec3f specular_s = L_s(scene, w_i, w_o, E, n, scene.meshes[closestMesh].material_id);
	                    //shading = add(shading, L_m(camera, scene, r, intersectionPoint, n, scene.meshes[closestMesh].material_id, depth));

	                    shading = add(add(diffuse_s,shading),specular_s); 
	                    
	                   
	                }
	                
	                
                }

                shading = add(shading, L_m(camera, scene, w_o, intersectionPoint, n, scene.meshes[closestMesh].material_id, depth));
            }
            else if(closestSphere != -1) // the closest object that intersects with the ray  a sphere
            {

                shading = L_a(scene.spheres[closestSphere].material_id, scene);
               
                for(int l = 0; l < scene.point_lights.size(); l++){
                    w_i = substract(intersectionPoint, scene.point_lights[l].position);
                    double distance = length(w_i);
                    I = scene.point_lights[l].intensity; 
                    E = E_i(I, w_i,distance);
                    w_o = substract(intersectionPoint,r.a);
                    w_i = normalize(w_i);
                    w_o = normalize(w_o);
                    if(shadow(scene, w_i, intersectionPoint, scene.point_lights[l].position)){ //if there is a shadow
                        continue;
                    }
                    else{

                        Vec3f diffuse_s = L_d(scene, w_i, E, n, scene.spheres[closestSphere].material_id);
                        Vec3f specular_s = L_s(scene, w_i, w_o, E, n, scene.spheres[closestSphere].material_id);
                        //shading = add(shading, L_m(camera, scene, r, intersectionPoint, n, scene.spheres[closestSphere].material_id, depth));
                        shading = add(add(diffuse_s,shading),specular_s); 
                        
                        
                    }

                    
                }

                shading = add(shading, L_m(camera, scene, w_o, intersectionPoint, n, scene.spheres[closestSphere].material_id, depth));
            }
            else
            {
                /* background color */
                shading.x = scene.background_color.x;
                shading.y = scene.background_color.y;
                shading.z = scene.background_color.z;
            }

            
    }

        return shading;

}

void produceImage(Scene scene, Camera cam, int width, int height, unsigned char* image){

	
	int p = 0;

	for(int y = 0; y < height; ++y){

		for(int x = 0; x < width; ++x){

			Ray r = generateRay(x, y, cam);

			Vec3f pixelShading = trace(scene, cam, r, scene.max_recursion_depth + 1);

			pixelShading = clamb(pixelShading);

			image[p++] = pixelShading.x; 
			image[p++] = pixelShading.y; 
			image[p++] = pixelShading.z; 

		}
	}

	write_ppm(cam.image_name.c_str(), image, width, height);
}

int main(int argc, char* argv[])
{
   
    // Sample usage for reading an XML scene file
    Scene scene;
    scene.loadFromXml(argv[1]);

    
    for(int i = 0; i < scene.cameras.size(); ++i){

    	int width = scene.cameras[i].image_width, height = scene.cameras[i].image_height;

    	unsigned char* image = new unsigned char [width * height * 3];

    	produceImage(scene, scene.cameras[i], width, height, image);
	}
    
  
    return 0; 
}
