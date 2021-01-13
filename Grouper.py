import random
from operator import itemgetter

import networkx as nx
import numpy as np
from numba import njit


def inRemaining(remaining_faces, c):
    if str(sorted(c)) in remaining_faces:
        return True
    return False

def removeRemaining(remaining_faces, min_mesh):
    remaining_faces.pop(str(sorted(min_mesh)))

def addRemaining(d,c):
    for min_mesh in c:
        srtd = sorted(min_mesh[0])
        d[str(srtd)] = srtd

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
        partitions_nv = []
        partitions_seed = []
        remaining_faces = {}
        for x in self.mesh_faces:
            srtd = sorted(x)
            remaining_faces[str(srtd)] = x

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

                if(len(candidates)>0):
                    recalc(candidates, current_nv)

                last_nv = self.getNV(current_chart[-1])
                expand_depth = 0
                a1 = adjacency_list[current_chart[-1][0]]
                a2 = adjacency_list[current_chart[-1][1]]
                a3 = adjacency_list[current_chart[-1][2]]
                temp = list(set.intersection(set(a1) , set(a2)))
                for y in temp:
                    c = [current_chart[-1][0], current_chart[-1][1], y]
                    srtd_c = sorted(c)
                    if str(srtd_c) not in added:
                        if inRemaining(remaining_faces, c):
                            x = remaining_faces[str(srtd_c)]
                            nv = self.getNV(x)
                            nvx = np.dot(nv, last_nv)
                            if (nvx <= 0):
                                continue
                            cost1 = costP1((np.array([self.mesh_vertices[x[0] - 1],
                                                      self.mesh_vertices[x[1] - 1],
                                                      self.mesh_vertices[x[2] - 1]])),
                                           np.array([self.mesh_vertices[current_chart[0][0] - 1],
                                                     self.mesh_vertices[current_chart[0][1] - 1],
                                                     self.mesh_vertices[current_chart[0][2] - 1]]))
                            cost2 = np.dot(nv, current_nv)
                            removeRemaining(remaining_faces, srtd_c)
                            candidates.append([x, cost1, cost1 * (1 - cost2), cost2, 1, nv])
                            self.expand(candidates, x, adjacency_list, 0, added, remaining_faces, current_nv, nv,
                                        expand_depth)

                temp = list(set.intersection(set(a2) , set(a3)))
                for y in temp:
                    c = [current_chart[-1][1], current_chart[-1][2], y]
                    srtd_c = sorted(c)
                    if str(srtd_c) not in added:
                        if inRemaining(remaining_faces, c):
                            x = remaining_faces[str(srtd_c)]
                            nv = self.getNV(x)
                            nvx = np.dot(nv, last_nv)
                            if (nvx <= 0):
                                continue
                            cost1 = costP1((np.array([self.mesh_vertices[x[0] - 1],
                                                      self.mesh_vertices[x[1] - 1],
                                                      self.mesh_vertices[x[2] - 1]])),
                                           np.array([self.mesh_vertices[current_chart[0][0] - 1],
                                                     self.mesh_vertices[current_chart[0][1] - 1],
                                                     self.mesh_vertices[current_chart[0][2] - 1]]))
                            cost2 = np.dot(nv, current_nv)
                            removeRemaining(remaining_faces, srtd_c)
                            candidates.append([x, cost1, cost1 * (1 - cost2), cost2, 1, nv])
                            self.expand(candidates, x, adjacency_list, 0, added, remaining_faces, current_nv, nv,
                                        expand_depth)

                temp = list(set.intersection(set(a1) , set(a3)))
                for y in temp:
                    c = [current_chart[-1][0], current_chart[-1][2], y]
                    srtd_c = sorted(c)
                    if str(srtd_c) not in added:
                        if inRemaining(remaining_faces, c):
                            x = remaining_faces[str(srtd_c)]
                            nv = self.getNV(x)
                            nvx = np.dot(nv, last_nv)
                            if (nvx <= 0):
                                continue
                            cost1 = costP1((np.array([self.mesh_vertices[x[0] - 1],
                                                      self.mesh_vertices[x[1] - 1],
                                                      self.mesh_vertices[x[2] - 1]])),
                                           np.array([self.mesh_vertices[current_chart[0][0] - 1],
                                                     self.mesh_vertices[current_chart[0][1] - 1],
                                                     self.mesh_vertices[current_chart[0][2] - 1]]))
                            cost2 = np.dot(nv, current_nv)
                            removeRemaining(remaining_faces, srtd_c)
                            candidates.append([x, cost1, cost1 * (1 - cost2), cost2, 1, nv])
                            self.expand(candidates, x, adjacency_list, 0, added, remaining_faces, current_nv, nv,
                                        expand_depth)

                if(len(candidates)==0):
                    break

                candidates = sorted(candidates, key=itemgetter(2))

                l = candidates

                min_mesh = l[0]
                min_angle = min_mesh[3]

                current_chart.append(min_mesh[0])
                added[str(sorted(min_mesh[0]))] = 1
                candidates.remove(min_mesh)
                nv = self.getNV(min_mesh[0])
                current_nv = np.array([(current_nv[0] + nv[0])/2, (current_nv[1] + nv[1])/2, (current_nv[2] + nv[2])/2])
                total += 1

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
                if(len(current_chart)<100 and len(partitions)>10):
                    min_id = -1
                    min_len = -1
                    min_nv = -1
                    cnt = -1
                    for x in partitions:
                        cnt+=1
                        s_tri = np.array([self.mesh_vertices[x[0][0]-1], self.mesh_vertices[x[0][0]-1], self.mesh_vertices[x[0][0]-1]])
                        d_tri = np.array([self.mesh_vertices[current_chart[0][0]-1], self.mesh_vertices[current_chart[0][0]-1], self.mesh_vertices[current_chart[0][0]-1]])
                        s_nv = partitions_nv[cnt]
                        d_nv = current_nv
                        angle = np.dot(s_nv,d_nv)
                        length = dist(cent(s_tri),cent(d_tri))
                        if(min_len == -1):
                            min_len = length*(1-angle)
                            min_id = cnt
                            min_nv = s_nv
                        elif length < min_len:
                            min_len = length*(1-angle)
                            min_id = cnt
                            min_nv = s_nv
                    partitions[min_id].extend(current_chart)
                    partitions_nv[min_id] = np.array([(current_nv[0] + min_nv[0])/2, (current_nv[1] + min_nv[1])/2, (current_nv[2] + min_nv[2])/2])
                else:
                    partitions.append(current_chart)
                    partitions_nv.append(current_nv)
                    partitions_seed.append(current_chart[0])

        #print("Total Partitions: " + str(len(partitions)))
        return partitions

    def getNV(self, face):
        mesh = np.array([self.mesh_vertices[face[0]-1], self.mesh_vertices[face[1]-1], self.mesh_vertices[face[2] -1] ])
        return surface_normal_newell(mesh)


    def expand(self, candidates, cd, adjacency_list, depth, added, remaining_faces, current_nv , last_nv, max_d):
        max_depth = max_d
        label = 1
        if(depth == max_depth):
            return
        if(depth+1 == max_depth):
            label = 0
        a1 = adjacency_list[cd[0]]
        a2 = adjacency_list[cd[1]]
        a3 = adjacency_list[cd[2]]
        c_v = np.array(
            [self.mesh_vertices[cd[0] - 1], self.mesh_vertices[cd[1] - 1],
             self.mesh_vertices[cd[2] - 1]])
        temp = list(set.intersection(set(a1), set(a2)))
        for y in temp:
            c = [cd[0], cd[1], y]
            sorted_c = sorted(c)
            if str(sorted_c) not in added:
                if inRemaining(remaining_faces, c):
                    x = remaining_faces[str(sorted_c)]
                    removeRemaining(remaining_faces, c)
                    nv = self.getNV(x)
                    cost1 = costP1((np.array([self.mesh_vertices[x[0] - 1],
                                              self.mesh_vertices[x[1] - 1],
                                              self.mesh_vertices[x[2] - 1]])),
                                   np.array([self.mesh_vertices[cd[0] - 1],
                                             self.mesh_vertices[cd[1] - 1],
                                             self.mesh_vertices[cd[2] - 1]]))
                    cost2 = np.dot(nv, current_nv)
                    nvx = np.dot(nv, last_nv)
                    candidates.append([x, cost1, cost1 * (1 - cost2), cost2, label, nv])
                    self.expand(candidates, x, adjacency_list, depth + 1, added, remaining_faces, current_nv, nv, max_d)

        temp = list(set.intersection(set(a2), set(a3)))
        for y in temp:
            c = [cd[1], cd[2], y]
            sorted_c = sorted(c)
            if str(sorted_c) not in added:
                if inRemaining(remaining_faces, c):
                    x = remaining_faces[str(sorted_c)]
                    removeRemaining(remaining_faces, c)
                    nv = self.getNV(x)
                    cost1 = costP1((np.array([self.mesh_vertices[x[0] - 1],
                                              self.mesh_vertices[x[1] - 1],
                                              self.mesh_vertices[x[2] - 1]])),
                                   np.array([self.mesh_vertices[cd[0] - 1],
                                             self.mesh_vertices[cd[1] - 1],
                                             self.mesh_vertices[cd[2] - 1]]))
                    cost2 = np.dot(nv, current_nv)
                    nvx = np.dot(nv, last_nv)
                    candidates.append([x, cost1, cost1 * (1 - cost2), cost2, label, nv])
                    self.expand(candidates, x, adjacency_list, depth + 1, added, remaining_faces, current_nv, nv, max_d)

        temp = list(set.intersection(set(a1), set(a3)))
        for y in temp:
            c = [cd[0], cd[2], y]
            sorted_c = sorted(c)
            if str(sorted_c) not in added:
                if inRemaining(remaining_faces, c):
                    x = remaining_faces[str(sorted_c)]
                    removeRemaining(remaining_faces, c)
                    nv = self.getNV(x)
                    cost1 = costP1((np.array([self.mesh_vertices[x[0] - 1],
                                              self.mesh_vertices[x[1] - 1],
                                              self.mesh_vertices[x[2] - 1]])),
                                   np.array([self.mesh_vertices[cd[0] - 1],
                                             self.mesh_vertices[cd[1] - 1],
                                             self.mesh_vertices[cd[2] - 1]]))
                    cost2 = np.dot(nv, current_nv)
                    nvx = np.dot(nv, last_nv)
                    candidates.append([x, cost1, cost1 * (1 - cost2), cost2, label, nv])
                    self.expand(candidates, x, adjacency_list, depth + 1, added, remaining_faces, current_nv, nv, max_d)


def recalc(candidates, current_nv):
    for count in range(len(candidates)):
        nv = candidates[count][5]
        cost2 = np.dot(nv, current_nv)
        candidates[count][3] = cost2
        candidates[count][2] = candidates[count][1] * (1 - cost2)


@njit
def costT(args1,args2,args3,results):
    for x in range(len(args1)):
        results[x] = (1 - np.dot(args1[x], surface_normal_newell(args2[x]))) * (dist(cent(args2[x]), cent(args3[x])))

@njit
def costP1(args1,args2):
    return dist(cent(args1), cent(args2))


@njit
def costP2(args1,args2,args3, results, r2):
    for x in range(len(args1)):
        a = np.dot(args1[x], surface_normal_newell(args2[x]))
        results[x] = (1 - a) * args3[x]
        r2[x] = a


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