import math
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.geometry.polygon import LinearRing

def intersection_over_union(target, list_of_bboxes, xp):
    print list_of_bboxes.shape
    target = LinearRing(get_vertex(target))
    list_of_bboxes = [LinearRing(get_vertex(i)) for i in list_of_bboxes]
    areas = xp.zeros(len(list_of_bboxes))
    for index, item in enumerate(list_of_bboxes):
        vertex = []
        if target.intersects(item):
            if isinstance(target.intersection(item),Point):
                continue
            for ver in target.intersection(item):
                if isinstance(ver,LineString):
                    u, v = ver.xy
                    vertex.append([u[0], v[0]])
                    vertex.append([u[1], v[1]])
                else:
                    vertex.append([ver.x, ver.y])
            for ver in target.coords[:4]:
                if Point(ver).within(Polygon(item)):
                    vertex.append(list(ver))
            for ver in item.coords[:4]:
                if Point(ver).within(Polygon(target)):
                    vertex.append(list(ver))
            areas[index] = Polygon(PolygonSort(vertex)).area
    return areas

def IoU(bbox_a,bbox_b):
    bbox_a = LinearRing(get_vertex(bbox_a))
    bbox_b = LinearRing(get_vertex(bbox_b))
    vertex = []
    if bbox_a.intersects(bbox_b):
        if isinstance(bbox_a.intersection(bbox_b),Point):
            return 0
        for ver in bbox_a.intersection(bbox_b):
            if isinstance(ver,LineString):
                u, v = ver.xy
                vertex.append([u[0], v[0]])
                vertex.append([u[1], v[1]])
            else:
                vertex.append([ver.x, ver.y])
        for ver in bbox_a.coords[:4]:
            if Point(ver).within(Polygon(bbox_b)):
                vertex.append(list(ver))
        for ver in bbox_b.coords[:4]:
            if Point(ver).within(Polygon(bbox_a)):
                vertex.append(list(ver))
        return Polygon(PolygonSort(vertex)).area
    return 0

def PolygonSort(corners):
    # calculate centroid of the polygon
    n = len(corners) # of corners
    cx = float(sum(x for x, y in corners)) / n
    cy = float(sum(y for x, y in corners)) / n
    # create a new list of corners which includes angles
    cornersWithAngles = []
    for x, y in corners:
        an = (math.atan2(y - cy, x - cx) + 2.0 * math.pi) % (2.0 * math.pi)
        cornersWithAngles.append((x, y, an))
    # sort it using the angles
    cornersWithAngles.sort(key = lambda tup: tup[2])
    # return the sorted corners w/ angles removed
    return map(lambda (x, y, an): (x, y), cornersWithAngles)


def get_vertex(bbox):
    x1 = math.cos(bbox[4]) * (-bbox[2]/2) - math.sin(bbox[4]) * (-bbox[3]/2) + bbox[0]
    y1 = math.sin(bbox[4]) * (-bbox[2] / 2) + math.cos(bbox[4]) * (-bbox[3] / 2) + bbox[1]
    x2 = math.cos(bbox[4]) * (bbox[2] / 2) - math.sin(bbox[4]) * (-bbox[3] / 2) + bbox[0]
    y2 = math.sin(bbox[4]) * (bbox[2] / 2) + math.cos(bbox[4]) * (-bbox[3] / 2) + bbox[1]
    x3 = math.cos(bbox[4]) * (bbox[2] / 2) - math.sin(bbox[4]) * (bbox[3] / 2) + bbox[0]
    y3 = math.sin(bbox[4]) * (bbox[2] / 2) + math.cos(bbox[4]) * (bbox[3] / 2) + bbox[1]
    x4 = math.cos(bbox[4]) * (-bbox[2] / 2) - math.sin(bbox[4]) * (bbox[3] / 2) + bbox[0]
    y4 = math.sin(bbox[4]) * (-bbox[2] / 2) + math.cos(bbox[4]) * (bbox[3] / 2) + bbox[1]

    return (x1,y1), (x2,y2), (x3,y3), (x4,y4)

if __name__ == '__main__':
    # sample1 = [16,16,8,4, math.pi/4]
    # polygon1 = LinearRing(get_vertex(sample1))
    # sample2 = [20, 20, 8, 8, 0]
    # polygon2 = LinearRing(get_vertex(sample2))
    #
    # intersec = polygon1.intersection(polygon2)
    # for i in polygon1.coords:
    #     print list(i)
    # point =  Point(20,20)
    # print Point(polygon1.coords[3]).within(Polygon(polygon2))
    bbox1 = np.array([20,20,8,8,0])
    bbox2 = np.array([[20,20,8,8,math.pi/3],[16,16,8,8,math.pi/4],[22,22,4,8, math.pi/6],[24,24,8,8,0]])


    areas = intersection_over_union(bbox1,bbox2, np)
    print areas
    # sample = np.array([[16,16,8,4, math.pi/4]])
    # for i in sample:
    #     print get_vertex(i)



