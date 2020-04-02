/*
*	This file is dependent on github open-source. All rights are perserved by original authors.
* 	Web: https://github.com/abreheret/polygon-intersection
*/
// Update: Add the API for EllipsoidSLAM.

#ifndef __EllipsoidSLAM_POLYGON_HPP__
#define __EllipsoidSLAM_POLYGON_HPP__

#include <cxcore.h>

namespace EllipsoidSLAM
{
	
float distPoint( CvPoint p1, CvPoint p2 ) ;
float distPoint(CvPoint2D32f p1,CvPoint2D32f p2) ;
bool segementIntersection(CvPoint p0_seg0,CvPoint p1_seg0,CvPoint p0_seg1,CvPoint p1_seg1,CvPoint * intersection) ;
bool segementIntersection(CvPoint2D32f p0_seg0,CvPoint2D32f p1_seg0,CvPoint2D32f p0_seg1,CvPoint2D32f p1_seg1,CvPoint2D32f * intersection) ;

bool pointInPolygon(CvPoint p,const CvPoint * points,int n) ;
bool pointInPolygon(CvPoint2D32f p,const CvPoint2D32f * points,int n) ;


#define MAX_POINT_POLYGON 64
struct Polygon {
	CvPoint pt[MAX_POINT_POLYGON];
	int     n;

	Polygon(int n_ = 0 ) { assert(n_>= 0 && n_ < MAX_POINT_POLYGON); n = n_;}
	virtual ~Polygon() {}

	void clear() { n = 0; }
	void add(const CvPoint &p) {if(n < MAX_POINT_POLYGON) pt[n++] = p;}
	void push_back(const CvPoint &p) {add(p);}
	int size() const { return n;}
	CvPoint getCenter() const ;
	const CvPoint & operator[] (int index) const { assert(index >= 0 && index < n); return pt[index];}
	CvPoint& operator[] (int index) { assert(index >= 0 && index < n); return pt[index]; }
	void pointsOrdered() ;
	float area() const ;
	bool pointIsInPolygon(CvPoint p) const ;
};


void intersectPolygon( const CvPoint * poly0, int n0,const CvPoint * poly1,int n1, Polygon & inter ) ;
void intersectPolygon( const Polygon & poly0, const Polygon & poly1, Polygon & inter ) ;
void intersectPolygonSHPC(const Polygon * sub,const Polygon* clip,Polygon* res) ;
void intersectPolygonSHPC(const Polygon & sub,const Polygon& clip,Polygon& res) ;

}

#endif //