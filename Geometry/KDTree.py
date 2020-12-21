import numpy as np
import scipy
from numba import jit
from scipy import spatial


class triangleTree:

    Tree = spatial.cKDTree

    triangles = []
    cent = []

    def addTriangles(self, triangles):
        self.triangles = triangles
        for x in self.triangles:
            c = calculate_centre_triangle(x)
            self.cent.append(c)


    def buildTree(self):
        self.cent = np.array(self.cent, dtype=np.float32)
        self.Tree = spatial.cKDTree(self.cent)

    def getpoints_within_d(self,triangle, d):
        c = self.Tree.query_ball_point(x=calculate_centre_triangle(triangle), r=d)
        points = []
        for x in c:
            t = self.triangles[x]
            if(np.array_equal(t,triangle)):
                continue
            points.append(t)

        return points

    def getpoints_between_range(self,triangle, d1,d2):
        c = self.Tree.query_ball_point(x=calculate_centre_triangle(triangle), r=d1)
        points1 = []
        for x in c:
            t = self.triangles[x]
            if(np.array_equal(t,triangle)):
                continue
            points1.append(t)

        c = self.Tree.query_ball_point(x=calculate_centre_triangle(triangle), r=d2)
        points2 = []
        for x in c:
            t = self.triangles[x]
            if(np.array_equal(t,triangle)):
                continue
            points2.append(t)

        all = np.concatenate([points1, points2],axis=0)
        unique, count = np.unique(all, return_index=True, axis=0)
        new_unique = []
        for x in range(len(unique)):
            if(count[x]==0):
                new_unique.append(unique[x])
        return(np.array(new_unique,dtype=np.float32))

@jit(nopython=True)
def calculate_centre_triangle(triangle):
    c = np.zeros(3, dtype=np.float32)
    c[0] = (triangle[0][0] + triangle[1][0] + triangle[2][0])/3
    c[1] = (triangle[0][1] + triangle[1][1] + triangle[2][1])/3
    c[2] = (triangle[0][2] + triangle[1][2] + triangle[2][2])/3
    return c