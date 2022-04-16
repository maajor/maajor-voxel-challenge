from scene import Scene
import taichi as ti
from taichi.math import *
scene = Scene(exposure=1)
scene.set_floor(-64, (1.0, 1.0, 1.0))
scene.set_background_color((0, 0, 0))
scene.set_directional_light((-1, 1, 0.5), 0.1, (1, 1, 1))
@ti.func
def rgbTo01(r,g,b):
    return vec3(r/255.0, g/255.0, b/255.0)
@ti.func
def proj_plane(o, n, t, p): 
    y = dot(p-o,n);xz=p-(o+n*y);bt=cross(t,n);return vec3(dot(xz,t), y, dot(xz, bt))
@ti.func
def ellipsoid(rx,ry,rz,p_unuse,pos):
    r = pos/vec3(rx,ry,rz); return ti.sqrt(dot(r,r))<1
@ti.func
def cylinder(r1,h,r2,cone,pos):
    r = vec2(pos.x/r1,pos.z/r2); return ti.sqrt(dot(r,r)) < mix(1.0-cone, 1.0, float(h-pos.y)*0.5/h) and pos.y<h and pos.y>-h
@ti.func
def box(x,y,z,round,pos):
    return pos.x<x and pos.x>-x and pos.z<z and pos.z>-z and pos.y<y and pos.y>-y
@ti.func
def make_implicit(func: ti.template(), p1, p2, p3, p4, pos, dir, up, color, mat, mode):
    max_r = 2 * int(max(p3,max(p1, p2))); dir = normalize(dir); up = normalize(cross(cross(dir, up), dir))
    for i,j,k in ti.ndrange((-max_r,max_r),(-max_r,max_r),(-max_r,max_r)): 
        xyz = proj_plane(vec3(0.0,0.0,0.0), dir, up, vec3(i,j,k))
        if func(p1,p2,p3,p4, xyz):
            if mode == 0: # additive
                scene.set_voxel(pos + vec3(i,j,k), mat, color)
            if mode == 1: # subtractive
                scene.set_voxel(pos + vec3(i,j,k), 0, color)
            if mode == 2:
                mat, c = scene.get_voxel(pos + vec3(i,j,k)) # rep, only color
                if mat > 0:
                    scene.set_voxel(pos + vec3(i,j,k), mat, color)
@ti.kernel
def initialize_voxels():
    make_implicit(ellipsoid, 35.9, 41.8, 34.4, 0.0, vec3(-16,8,-22), vec3(0.3,1.0,-0.1), vec3(0.7,-0.3,-0.7), rgbTo01(255,217,187), 1, 0)
    make_implicit(cylinder, 33.8, 10.9, 32.9, 0.2, vec3(-6,30,-23), vec3(0.4,0.9,-0.1), vec3(0.9,-0.4,0.0), rgbTo01(114,161,255), 1, 0)
    make_implicit(ellipsoid, 11.1, 10.6, 2.9, 0.0, vec3(-23,-17,8), vec3(0.4,0.8,0.6), vec3(0.9,-0.4,-0.1), rgbTo01(255,141,143), 1, 2)
    make_implicit(ellipsoid, 10.2, 7.7, 15.3, 0.0, vec3(-45,-4,-18), vec3(-0.4,0.9,0.2), vec3(0.1,-0.2,1.0), rgbTo01(255,141,143), 1, 2)
    make_implicit(ellipsoid, 9.6, 7.2, 6.2, 0.0, vec3(-27,3,12), vec3(0.2,1.0,0.2), vec3(0.9,-0.3,0.2), rgbTo01(0,0,0), 1, 2)
    make_implicit(ellipsoid, 8.3, 6.5, 8.6, 0.0, vec3(-26,3,11), vec3(0.3,1.0,0.1), vec3(0.9,-0.3,0.2), rgbTo01(255,255,255), 1, 2)
    make_implicit(ellipsoid, 4.1, 4.7, 2.3, 0.0, vec3(-25,3,12), vec3(0.2,1.0,0.1), vec3(0.9,-0.3,0.3), rgbTo01(12,41,60), 1, 0)
    make_implicit(ellipsoid, 9.9, 6.9, 3.6, 0.0, vec3(-51,9,-11), vec3(0.3,0.9,0.1), vec3(0.2,-0.2,1.0), rgbTo01(0,0,0), 1, 2)
    make_implicit(ellipsoid, 8.7, 5.3, 12.1, 0.0, vec3(-47,8,-12), vec3(0.3,0.9,0.1), vec3(0.2,-0.2,1.0), rgbTo01(255,255,255), 1, 2)
    make_implicit(ellipsoid, 4.1, 4.7, 1.5, 0.0, vec3(-49,8,-7), vec3(0.2,1.0,0.1), vec3(0.4,-0.2,0.9), rgbTo01(12,41,60), 1, 0)
    make_implicit(box, 5.3, 3.9, 1.0, 0.0, vec3(-19,13,12), vec3(0.8,-0.5,0.2), vec3(-0.2,0.1,1.0), rgbTo01(0,0,0), 1, 2)
    make_implicit(cylinder, 12.2, 16.2, 9.7, 0.4, vec3(-17,-44,-25), vec3(-0.0,1.0,0.0), vec3(0.8,0.0,-0.7), rgbTo01(255,212,195), 1, 0)
    make_implicit(cylinder, 36.8, 8.6, 36.8, 0.2, vec3(-20,-50,-18), vec3(-0.1,1.0,0.0), vec3(0.9,0.1,-0.5), rgbTo01(140,123,17), 1, 0)
    make_implicit(cylinder, 27.4, 3.1, 26.6, 0.0, vec3(-20,-41,-18), vec3(-0.0,1.0,-0.0), vec3(0.8,-0.0,-0.7), rgbTo01(255,255,255), 1, 0)
    make_implicit(cylinder, 27.4, 3.1, 26.6, 0.0, vec3(-20,-41,-18), vec3(-0.0,1.0,-0.0), vec3(0.8,-0.0,-0.7), rgbTo01(255,255,255), 1, 0)
    make_implicit(box, 3.8, 6.4, 1.2, 0.0, vec3(-45,20,-8), vec3(0.2,-0.1,1.0), vec3(-0.9,0.4,0.3), rgbTo01(0,0,0), 1, 2)
    make_implicit(cylinder, 25.7, 9.1, 26.2, 0.6, vec3(0,43,-25), vec3(0.4,0.9,-0.1), vec3(0.9,-0.4,0.0), rgbTo01(142,134,57), 1, 0)
    make_implicit(ellipsoid, 11.6, 9.1, 12.7, 0.0, vec3(3,48,-26), vec3(0.4,0.9,-0.1), vec3(0.9,-0.4,0.0), rgbTo01(142,134,57), 1, 0)
    make_implicit(cylinder, 10.6, 41.6, 11.3, 0.4, vec3(34,-19,10), vec3(-0.1,1.0,-0.2), vec3(1.0,0.0,-0.1), rgbTo01(142,134,57), 1, 0)
    make_implicit(cylinder, 8.0, 16.1, 8.1, 0.1, vec3(29,32,-3), vec3(-0.2,0.9,-0.3), vec3(1.0,0.1,-0.2), rgbTo01(142,134,57), 1, 0)
    make_implicit(cylinder, 8.0, 16.1, 8.1, 0.1, vec3(18,49,-13), vec3(-0.8,0.5,-0.4), vec3(0.6,0.7,-0.4), rgbTo01(142,134,57), 1, 0)
    make_implicit(cylinder, 8.0, 16.1, 8.1, 0.0, vec3(1,49,-25), vec3(-0.6,-0.5,-0.6), vec3(0.6,0.2,-0.8), rgbTo01(142,134,57), 1, 0)
    make_implicit(cylinder, 10.6, 44.9, 11.3, 0.4, vec3(47,-13,7), vec3(-0.1,1.0,-0.2), vec3(1.0,0.0,-0.1), rgbTo01(142,134,57), 1, 0)
    make_implicit(cylinder, 8.0, 16.1, 8.1, 0.6, vec3(39,39,-6), vec3(-0.4,0.9,-0.3), vec3(0.9,0.3,-0.3), rgbTo01(142,134,57), 1, 0)
    make_implicit(cylinder, 5.3, 16.1, 5.3, 1.0, vec3(22,56,-16), vec3(-0.8,0.4,-0.5), vec3(0.6,0.7,-0.4), rgbTo01(142,134,57), 1, 0)
    make_implicit(ellipsoid, 2.0, 10.2, 7.5, 0.0, vec3(11,-1,1), vec3(0.6,0.8,0.1), vec3(0.6,-0.4,-0.7), rgbTo01(255,217,187), 1, 0)
    make_implicit(ellipsoid, 3.7, 3.7, 3.7, 0.0, vec3(11,-19,4), vec3(0.4,0.9,-0.1), vec3(0.6,-0.4,-0.7), rgbTo01(255,255,255), 1, 0)
    make_implicit(cylinder, 1.0, 8.8, 1.0, 0.0, vec3(11,-12,4), vec3(0.0,1.0,-0.1), vec3(0.0,-0.1,-1.0), rgbTo01(255,255,255), 1, 0)
    make_implicit(cylinder, 34.7, 8.6, 35.9, 0.1, vec3(-10,24,-22), vec3(0.4,0.9,-0.1), vec3(0.9,-0.4,0.0), rgbTo01(114,161,255), 1, 0)
    make_implicit(box, 5.3, 5.9, 1.0, 0.0, vec3(-26,15,10), vec3(0.9,-0.0,0.3), vec3(-0.3,0.1,0.9), rgbTo01(0,0,0), 1, 2)
    make_implicit(box, 3.8, 8.1, 1.0, 0.0, vec3(-48,20,-17), vec3(0.3,0.2,0.9), vec3(-0.9,0.3,0.2), rgbTo01(0,0,0), 1, 2)
    make_implicit(box, 1.1, 2.0, 1.1, 0.0, vec3(-40,-21,-7), vec3(0.7,-0.5,0.5), vec3(0.1,-0.5,-0.8), rgbTo01(255,54,58), 1, 0)
    make_implicit(box, 1.1, 3.2, 1.1, 0.0, vec3(-39,-20,-3), vec3(-0.1,0.6,0.8), vec3(0.8,-0.5,0.4), rgbTo01(255,54,58), 1, 0)
    make_implicit(box, 4.6, 17.5, 6.0, 0.0, vec3(-36,5,-5), vec3(0.3,0.9,0.0), vec3(0.9,-0.3,0.0), rgbTo01(255,70,70), 1, 0)
    make_implicit(box, 5.3, 3.9, 1.0, 0.0, vec3(-14,10,13), vec3(0.7,-0.7,0.2), vec3(-0.2,0.1,1.0), rgbTo01(0,0,0), 1, 2)
initialize_voxels()
scene.finish()