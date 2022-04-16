from scene import Scene
import taichi as ti
from taichi.math import *
scene = Scene(exposure=1)
scene.set_floor(-64, (1.0, 1.0, 1.0))
scene.set_background_color((0, 0, 0))
scene.set_direction_light((-1, 1, 1), 0.1, (1, 1, 1))
@ti.func
def rgbTo01(r,g,b):
    return vec3(r/255.0, g/255.0, b/255.0)
@ti.func 
def proj_ray(origin, ray, pos): d = dot(pos-origin, ray); return vec2(d, distance(pos, origin+ray*d))
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
def tri_prim(r, h, p_unuse1, p_unuse2, pos):
    return pos.z>0 and (pos.x*1.732+pos.z) < r and (-pos.x*1.732+pos.z) < r and pos.y<h and pos.y>0
@ti.func
def make_implicit(func: ti.template(), p1, p2, p3, p4, pos, dir, up, color, mode):
    max_r = 2 * int(max(p3,max(p1, p2))); dir = normalize(dir); up = normalize(cross(cross(dir, up), dir))
    for i,j,k in ti.ndrange((-max_r,max_r),(-max_r,max_r),(-max_r,max_r)): 
        xyz = proj_plane(vec3(0,0,0), dir, up, vec3(i,j,k))
        if func(p1,p2,p3,p4, xyz):
            scene.set_voxel(pos + vec3(i,j,k), mode, color)
@ti.kernel
def initialize_voxels():
    # Your code here! :-)
    make_implicit(ellipsoid, 30, 40, 30, 0.0, vec3(0,20,0), vec3(1,3,0), vec3(0,1,0), rgbTo01(211,174,168), 1)


initialize_voxels()

scene.finish()
