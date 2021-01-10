import random
from operator import itemgetter

import networkx as nx
import numpy as np
from numba import jit, njit
from numba.typed import List
import pickle as pkl


def inRemaining(remaining_faces, c):
    if str([c[0],c[1],c[2]]) in remaining_faces:
        return True
    elif str([c[0],c[2],c[1]]) in remaining_faces:
        return True
    elif str([c[1],c[0],c[2]]) in remaining_faces:
        return True
    elif str([c[1],c[2],c[0]]) in remaining_faces:
        return True
    elif str([c[2],c[1],c[0]]) in remaining_faces:
        return True
    elif str([c[2],c[0],c[1]]) in remaining_faces:
        return True
    return False

def removeRemaining(remaining_faces, min_mesh):
    try:
        remaining_faces.pop(str([min_mesh[0], min_mesh[1], min_mesh[2]]))
    except:
        pass
    try:
        remaining_faces.pop(str([min_mesh[0], min_mesh[2], min_mesh[1]]))
    except:
        pass
    try:
        remaining_faces.pop(str([min_mesh[1], min_mesh[0], min_mesh[2]]))
    except:
        pass
    try:
        remaining_faces.pop(str([min_mesh[1], min_mesh[2], min_mesh[0]]))
    except:
        pass
    try:
        remaining_faces.pop(str([min_mesh[2], min_mesh[1], min_mesh[0]]))
    except:
        pass
    try:
        remaining_faces.pop(str([min_mesh[2], min_mesh[0], min_mesh[1]]))
    except:
        pass

def addRemaining(d,c):
    for min_mesh in c:
        d[str(min_mesh[0])] = min_mesh[0]

class partition():

    mesh_faces = None
    mesh_vertices = None
    initialParts = 8
    numParts = None

    def __init__(self, mesh_vertices, mesh_faces, parts):
        self.mesh_faces = mesh_faces
        self.mesh_vertices = mesh_vertices
        self.initialParts = parts


    def getParts(self):

        partitions = []

        remaining_faces = {}
        for x in self.mesh_faces:
            remaining_faces[str(x)] = x

        g = nx.Graph()

        for x in self.mesh_faces:
            g.add_edge(x[0], x[1])
            g.add_edge(x[1], x[2])
            g.add_edge(x[2], x[0])

        adjacency_list = nx.to_dict_of_lists(g)

        part_size = (len(self.mesh_faces)/self.initialParts)
        added = {}
        total = 0
        while(len(remaining_faces)>0):

            current_chart = []
            if(len(remaining_faces)<=0):
                break
            rand_face = random.randint(0, len(remaining_faces)-1)

            while(str(sorted(remaining_faces[list(remaining_faces.keys())[rand_face]])) in added):
                rand_face = random.randint(0, len(remaining_faces)-1)

            current_chart.append(remaining_faces[list(remaining_faces.keys())[rand_face]])
            added[str(sorted(remaining_faces[list(remaining_faces.keys())[rand_face]]))] = 1

            current_nv = self.getNV(current_chart[-1])
            remaining_faces.pop(list(remaining_faces.keys())[rand_face])

            if(len(remaining_faces)<=0):
                break

            done = 0
            candidates = []
            while(done==0):

                min_score = None
                min_mesh = None
                min_angle = None
                a1 = adjacency_list[current_chart[-1][0]]
                a2 = adjacency_list[current_chart[-1][1]]
                a3 = adjacency_list[current_chart[-1][2]]

                temp = list(set.intersection(set(a1) , set(a2)))
                for y in temp:
                    c = [current_chart[-1][0], current_chart[-1][1], y]
                    if str(sorted(c)) not in added:
                        x = c
                        cost1 = costP1(
                            (np.array([self.mesh_vertices[x[0]-1], self.mesh_vertices[x[1]-1], self.mesh_vertices[x[2]-1]])),
                            np.array([self.mesh_vertices[current_chart[0][0]-1], self.mesh_vertices[current_chart[0][1]-1],
                                      self.mesh_vertices[current_chart[0][2]-1] ]))
                        if inRemaining(remaining_faces, c):
                            removeRemaining(remaining_faces, c)
                            candidates.append([c, cost1])

                temp = list(set.intersection(set(a2) , set(a3)))
                for y in temp:
                    c = [current_chart[-1][1], current_chart[-1][2], y]
                    if str(sorted(c)) not in added:
                        x = c
                        cost1 = costP1((np.array([self.mesh_vertices[x[0]-1], self.mesh_vertices[x[1]-1], self.mesh_vertices[x[2]-1]])),np.array([self.mesh_vertices[current_chart[0][0]-1], self.mesh_vertices[current_chart[0][1]-1],self.mesh_vertices[current_chart[0][2]-1]]))
                        if inRemaining(remaining_faces, c):
                            removeRemaining(remaining_faces, c)
                            candidates.append([c, cost1])

                temp = list(set.intersection(set(a1) , set(a3)))
                for y in temp:
                    c = [current_chart[-1][0], current_chart[-1][2], y]
                    if str(sorted(c)) not in added:
                        x = c
                        cost1 = costP1((np.array([self.mesh_vertices[x[0]-1],self.mesh_vertices[x[1]-1],self.mesh_vertices[x[2]-1]])),np.array([self.mesh_vertices[current_chart[0][0]-1], self.mesh_vertices[current_chart[0][1]-1], self.mesh_vertices[current_chart[0][2]-1]]))
                        if inRemaining(remaining_faces,c):
                            removeRemaining(remaining_faces,c)
                            candidates.append([c,cost1])

                if(len(candidates)==0):
                    break

                args1 = List()
                args2 = List()
                args3 = List()

                l = sorted(candidates, key=itemgetter(1))

                if(len(l)>30):
                    l = l[:30]

                for y in l:
                    x = y[0]
                    args2.append(np.array([self.mesh_vertices[x[0] -1 ],self.mesh_vertices[x[1] -1 ],self.mesh_vertices[x[2] -1] ]))
                    args3.append(y[1])
                    args1.append(np.array(current_nv))

                results = List()
                results2 = List()
                for x in range(len(args1)):
                    results.append(10.0)
                    results2.append(10.0)
                costP2(args1,args2,args3, results, results2)

                i = 0
                for cost in results:
                    if min_score is None:
                        min_score = cost
                        min_mesh = l[i]
                        min_angle = results2[i]
                    elif cost < min_score:
                        min_angle = results2[i]
                        min_score = cost
                        min_mesh = l[i]
                    i+=1

                current_chart.append(min_mesh[0])
                added[str(sorted(min_mesh[0]))] = 1

                candidates.remove(min_mesh)

                nv = self.getNV(min_mesh[0])
                current_nv = np.array([(current_nv[0] + nv[0])/2, (current_nv[1] + nv[1])/2, (current_nv[2] + nv[2])/2])

                total += 1
                # if (total >= 1000 and total % 1000 == 0):
                #     print(str(total) + " " + str(len(remaining_faces)))

                if(len(current_chart)>=part_size):
                    print(str(len(partitions)+1)+".1 " + str(len(current_chart)))
                    addRemaining(remaining_faces, candidates)
                    done=1
                elif(len(remaining_faces)<=0 and len(candidates)==0):
                    print(str(len(partitions)+1)+".2 " + str(len(current_chart)))
                    done=1
                elif(len(candidates)==0):
                    print(str(len(partitions)+1)+".3 " + str(len(current_chart)))
                    done=1

            if(len(current_chart)>0):
                partitions.append(current_chart)

        pkl.dump(partitions, open("partitions.p", "wb"))
        print("Total Partitions: " + str(len(partitions)))
        return partitions

    def getNV(self, face):
        mesh = np.array([self.mesh_vertices[face[0]-1], self.mesh_vertices[face[1]-1], self.mesh_vertices[face[2] -1] ])
        return surface_normal_newell(mesh)


@njit
def costT(args1,args2,args3,results):
    for x in range(len(args1)):
        results[x] = (1 - np.dot(args1[x], surface_normal_newell(args2[x]))) * (dist(cent(args2[x]), cent(args3[x])))

@njit
def costP1(args1,args2):
    return dist(cent(args1), cent(args2))



@njit
def costP2(args1,args2,args3,results, r2):
    for x in range(len(args1)):
        a = np.dot(args1[x], surface_normal_newell(args2[x]))
        results[x] = (1 - a) * args3[x]
        r2[x] = 1 - a


@njit
def surface_normal_newell(poly):

    n = np.array([0.0, 0.0, 0.0])

    for i, v_curr in enumerate(poly):
        v_next = poly[(i+1) % len(poly), :]
        n[0] += (v_curr[1] - v_next[1]) * (v_curr[2] + v_next[2])
        n[1] += (v_curr[2] - v_next[2]) * (v_curr[0] + v_next[0])
        n[2] += (v_curr[0] - v_next[0]) * (v_curr[1] + v_next[1])

    n = n / np.linalg.norm(n)
    return n


@njit
def dist(c,y):
    return np.sqrt(np.square(y[0] - c[0]) + np.square(y[1] - c[1]) + np.square(y[2] - c[2]))


@njit
def cent(triangle):
    c = np.zeros(3)
    c[0] = (triangle[0][0] + triangle[1][0] + triangle[2][0])/3
    c[1] = (triangle[0][1] + triangle[1][1] + triangle[2][1])/3
    c[2] = (triangle[0][2] + triangle[1][2] + triangle[2][2])/3
    return c