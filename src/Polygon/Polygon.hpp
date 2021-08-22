/*
*	This file is dependent on github open-source. All rights are perserved by original authors.
* 	Web: https://github.com/abreheret/polygon-intersection
*/
// Update: Add the API for EllipsoidSLAM.

#ifndef __EllipsoidSLAM_POLYGON_HPP__
#define __EllipsoidSLAM_POLYGON_HPP__

// #include <cxcore.h>  // not recoginized by OpenCV 4.x
#include <opencv2/core.hpp>

namespace EllipsoidSLAM
{
float distPoint( cv::Point2d p1, cv::Point2d p2 ) ;
bool segementIntersection(cv::Point2d p0_seg0,cv::Point2d p1_seg0,cv::Point2d p0_seg1,cv::Point2d p1_seg1,cv::Point2d * intersection) ;

bool pointInPolygon(cv::Point2d p,const cv::Point2d * points,int n) ;


#define MAX_POINT_POLYGON 64
struct Polygon {
	cv::Point2d pt[MAX_POINT_POLYGON];
	int     n;

	Polygon(int n_ = 0 ) { assert(n_>= 0 && n_ < MAX_POINT_POLYGON); n = n_;}
	virtual ~Polygon() {}

	void clear() { n = 0; }
	void add(const cv::Point2d &p) {if(n < MAX_POINT_POLYGON) pt[n++] = p;}
	void push_back(const cv::Point2d &p) {add(p);}
	int size() const { return n;}
	cv::Point2d getCenter() const ;
	const cv::Point2d & operator[] (int index) const { assert(index >= 0 && index < n); return pt[index];}
	cv::Point2d& operator[] (int index) { assert(index >= 0 && index < n); return pt[index]; }
	void pointsOrdered() ;
	float area() const ;
	bool pointIsInPolygon(cv::Point2d p) const ;
};


void intersectPolygon( const cv::Point2d * poly0, int n0,const cv::Point2d * poly1,int n1, Polygon & inter ) ;
void intersectPolygon( const Polygon & poly0, const Polygon & poly1, Polygon & inter ) ;
void intersectPolygonSHPC(const Polygon * sub,const Polygon* clip,Polygon* res) ;
void intersectPolygonSHPC(const Polygon & sub,const Polygon& clip,Polygon& res) ;

}

#endif //