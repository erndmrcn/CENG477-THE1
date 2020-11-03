#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <fstream>

#define ABS(a) ((a)>0?(a):-1*(a))

using namespace std;
class Vec3
{
private:
	double x,y,z;
public:
	// here comes methods
	double getX()
	{
		return x;
	}
	void setX(double a)
	{
		x = a;
	}
	double getY()
	{
		return y;
	}
	void setY(double b)
	{
		y = b;
	}
	double getZ(void)
	{
		return z;
	}
	void setZ(double c)
	{
		z = c;
	}
	double len(void)
	{
		return sqrt(x^2+y^2+z^2);
	}
	double len2(void)
	{
		return x^2+y^2+z^2;
	}

}

class Color
{
private:
	int r,g,b;
public:
	/* here comes methods */
	int getRed(void)
	{
		return r;
	}
	void setRed(int c)
	{
		r = c;
	}
	int getGreen(void)
	{
		return g;
	}
	void setGreen(int c)
	{
		g = c;
	}
	int getBlue(void)
	{
		return b;
	}
	void setBlue(int c)
	{
		b = c;
	}
}

class Ray
{
private:
	Vec3 a;
	Vec3 b;
public:
	// for methods 
	Vec3 getV1(void)
	{
		return a;
	}
	Vec3 getV1(void)
	{
		return b;
	}
	/* add set functions */
}

class Sphere
{
private:
	Vec3 center;
	double r; // radius
	Color color;
public:
	//methods
	Vec3 getCenter(void)
	{
		return center;
	}
	double getRadius(void)
	{
		return r;
	}
	Color getColor(void)
	{
		return color;
	}
}

class Triangle
{
private:
	Vec3 a,b,c;
	Color color;
public:
	/* methods */
	Vec3 getFirst(void)
	{
		return a;
	}
	Vec3 getSecond(void)
	{
		return b;
	}
	Vec3 getThird(void)
	{
		return c;
	}
	Color getColor(void)
	{
		return color;
	}
}

int numSpheres;
int numTriangles;

Sphere* spheres;
Triangle* triangles;

class Camera
{
private:
	Vec3 pos;
	Vec3 gaze;
	Vec3 v;
	Vec3 u;
	double l,r,b,t;
	double di;
public:
	/* methods */
	Vec3 getPos(void)
	{
		return pos;
	}
	Vec3 getGaze(void)
	{
		return gaze;
	}
	Vec3 getU(void)
	{
		return u;
	}
	Vec3 getV(void)
	{
		return v;
	}

	double getLeft(void)
	{
		return l;
	}
	double getRight(void)
	{
		return r;
	}
	double getTop(void)
	{
		return t;
	}
      	double getBottom(void)
	{
		return b;
	}
	double getDi(void)
	{
		return di;
	}
	/* write set functions for private data */
}

Camera cam;

int sizeX, sizeY;

double pixelW, pixelH;

double halfPixelW, halfPixelH;


Color **image;

Vec3 cross(Vec3 a, Vec3 b)
{
	Vec3 tmp;

	tmp.setX(a.getY()*b.getZ() - b.getY()*a.getX());
        tmp.setY(b.getX()*a.getZ() - a.getX()*b.getZ());
	tmp.setZ(a.getX()*b.getY() - b.getX()*a.getY());

	return tmp;	
}

double dot(Vec3 a, Vec3 b)
{
	return a.getX()*b.getX() + a.getY()*b.getY() + a.getZ()*b.getZ();
}

Vec3 normalize(Vec3 v)
{
	Vec3 tmp;
	double d;

	d = v.len();

	tmp.setX(v.getX()/d);
	tmp.setY(v.getY()/d);
	tmp.setZ(v.getZ()/d);
}

//
// add,mult,distance,equal methods into the Vec3 class
//

/* file operations read scene and camera files */

void readScene(Sring filename)
{

}

void readCamera(string filename)
{
	double gazex,gazey,gazez,posx,posy,posz,right,left,bottom,top,dimension,height,width;
	Vec3 v1,v2,v3;
	ifstream myfile open(filename, ios:in);
	myfile.open("simple_camera.txt");
	myfile >> posx >> posy >> posz;
	myfile >> gazex >> gazey >> gazez;
	myfile >> left >> right >> bottom >> top >> dimension >> width >> height;

	// continue from here //

}

int main(void) // for now void 
{
	// take input
	// read files
	// start image 2D grid
	// generate rays
	// check for intersects
	// write 2D grid

	// write an image file
	
}
