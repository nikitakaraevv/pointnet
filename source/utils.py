import plotly.graph_objects as go
import numpy as np
import scipy.spatial.distance
import math
import random

def cent_norm(arr_in):
    mean = np.mean(arr_in, axis=0)
    verts_centered = arr_in - mean
    max_point = 0
    for row in verts_centered:
        if (np.linalg.norm(row)> max_point):
            max_point = np.linalg.norm(row)
    verts_normalized = verts_centered/max_point   
    return verts_normalized

def generate_point(pt1, pt2, pt3):
    s, t = sorted([random.random(), random.random()])
    return (s * pt1[0] + (t-s)*pt2[0] + (1-t)*pt3[0],
            s * pt1[1] + (t-s)*pt2[1] + (1-t)*pt3[1],
            s * pt1[2] + (t-s)*pt2[2] + (1-t)*pt3[2])

def triangle_area(pnt1, pnt2, pnt3):
    side_a = np.linalg.norm(pnt1 - pnt2)
    side_b = np.linalg.norm(pnt2 - pnt3)
    side_c = np.linalg.norm(pnt3 - pnt1)
    s = 0.5 * ( side_a + side_b + side_c)
    return math.sqrt(max(s * (s - side_a) * (s - side_b) * (s - side_c),0))

def sample_points(verts, faces):
    areas = np.zeros((len(faces)))
    for i in range(len(areas)):
        areas[i] = triangle_area(verts[faces[i][0]],verts[faces[i][1]], verts[faces[i][2]])
    sample_faces = random.choices(faces, weights=areas,cum_weights=None, k=1024)
    sample_points = np.zeros((1024,3))
    for i in range(len(sample_faces)):
        sample_points[i] = generate_point(verts[sample_faces[i][0]],verts[sample_faces[i][1]], verts[sample_faces[i][2]] )
    return sample_points



def rotation_z(arr,theta):
    theta = theta * math.pi/180
    rot = np.array([[ math.cos(theta), -math.sin(theta), 0],
                   [ math.sin(theta), math.cos(theta), 0],
                   [0, 0, 1]])
    arr_rot = rot.dot(arr.transpose()).transpose()
    return arr_rot
    
    
def add_noise(arr):
    noise = np.random.normal(0,0.02,(arr.shape))
    return (arr+noise)

def pcshow(xs,ys,zs):
    fig = go.Figure(data=[go.Scatter3d(x=xs, y=ys, z=zs,
                                   mode='markers')])
    fig.update_traces(marker=dict(size=2,
                      line=dict(width=2,
                      color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()