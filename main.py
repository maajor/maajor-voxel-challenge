from scene import Scene
import taichi as ti
from taichi.math import *
scene = Scene(voxel_edges=0.01, exposure=1)
scene.set_floor(-64, (1.0, 1.0, 1.0))
scene.set_background_color((0, 0, 0))
scene.set_directional_light((-1, 1, 0.3), 0.0, (1, 1, 1))
@ti.func
def rgb(r,g,b):
    return vec3(r/255.0, g/255.0, b/255.0)
@ti.func
def proj_plane(o, n, t, p): 
    y = dot(p-o,n);xz=p-(o+n*y);bt=cross(t,n);return vec3(dot(xz,t), y, dot(xz, bt))
@ti.func
def elli(rx,ry,rz,p1_unused,p2_unused,p3_unused,p):
    r = p/vec3(rx,ry,rz); return ti.sqrt(dot(r,r))<1
@ti.func
def cyli(r1,h,r2,round, cone, hole_unused, p):
    ms=min(r1,min(h,r2));rr=ms*round;rt=mix(cone*(max(ms-rr,0)),0,float(h-p.y)*0.5/h);r=vec2(p.x/r1,p.z/r2)
    d=vec2((r.norm()-1.0)*ms+rt,ti.abs(p.y)-h)+rr; return min(max(d.x,d.y),0.0)+max(d,0.0).norm()-rr<0
@ti.func
def box(x, y, z, round, cone, unused, p):
    ms=min(x,min(y,z));rr=ms*round;rt=mix(cone*(max(ms-rr,0)),0,float(y-p.y)*0.5/y);q=ti.abs(p)-vec3(x-rt,y,z-rt)+rr
    return ti.max(q, 0.0).norm() + ti.min(ti.max(q.x, ti.max(q.y, q.z)), 0.0) - rr< 0
@ti.func
def tri(r1, h, r2, round_unused, cone, vertex, p):
    r = vec3(p.x/r1, p.y, p.z/r2);rt=mix(1.0-cone,1.0,float(h-p.y)*0.5/h);r.z+=(r.x+1)*mix(-0.577, 0.577, vertex)
    q = ti.abs(r); return max(q.y-h,max(q.z*0.866025+r.x*0.5,-r.x)-0.5*rt)< 0
@ti.func
def make(func: ti.template(), p1, p2, p3, p4, p5, p6, pos, dir, up, color, mat, mode):
    max_r = 2 * int(max(p3,max(p1, p2))); dir = normalize(dir); up = normalize(cross(cross(dir, up), dir))
    for i,j,k in ti.ndrange((-max_r,max_r),(-max_r,max_r),(-max_r,max_r)): 
        xyz = proj_plane(vec3(0.0,0.0,0.0), dir, up, vec3(i,j,k))
        if func(p1,p2,p3,p4,p5,p6,xyz):
            if mode == 0: scene.set_voxel(pos + vec3(i,j,k), mat, color) # additive
            if mode == 1: scene.set_voxel(pos + vec3(i,j,k), 0, color) # subtractive
            if mode == 2 and scene.get_voxel(pos + vec3(i,j,k))[0] > 0: scene.set_voxel(pos + vec3(i,j,k), mat, color)
@ti.kernel
def initialize_voxels():
    make(elli,35.9,41.8,34.4,0.0,0.0,0.0,vec3(-16,8,-22),vec3(0.3,1.0,-0.1),vec3(0.7,-0.3,-0.7),rgb(255,217,187),1,0)
    make(cyli,33.8,10.9,32.9,0.4,0.2,0.0,vec3(-6,30,-23),vec3(0.4,0.9,-0.1),vec3(0.9,-0.4,0.0),rgb(114,161,255),1,0)
    make(elli,11.1,10.6,2.9,0.0,0.0,0.0,vec3(-23,-17,8),vec3(0.4,0.8,0.6),vec3(0.9,-0.4,-0.1),rgb(255,141,143),1,2)
    make(elli,10.2,7.7,15.3,0.0,0.0,0.0,vec3(-45,-4,-18),vec3(-0.4,0.9,0.2),vec3(0.1,-0.2,1.0),rgb(255,141,143),1,2)
    make(elli,9.6,7.2,6.2,0.0,0.0,0.0,vec3(-27,3,12),vec3(0.2,1.0,0.2),vec3(0.9,-0.3,0.2),rgb(0,0,0),1,2)
    make(elli,8.3,6.5,8.6,0.0,0.0,0.0,vec3(-26,3,11),vec3(0.3,1.0,0.1),vec3(0.9,-0.3,0.2),rgb(255,255,255),1,2)
    make(elli,4.1,4.7,2.3,0.0,0.0,0.0,vec3(-25,3,12),vec3(0.2,1.0,0.1),vec3(0.9,-0.3,0.3),rgb(12,41,60),1,0)
    make(elli,9.9,6.9,3.6,0.0,0.0,0.0,vec3(-51,9,-11),vec3(0.3,0.9,0.1),vec3(0.2,-0.2,1.0),rgb(0,0,0),1,2)
    make(elli,8.7,5.3,12.1,0.0,0.0,0.0,vec3(-47,8,-12),vec3(0.3,0.9,0.1),vec3(0.2,-0.2,1.0),rgb(255,255,255),1,2)
    make(elli,4.1,4.7,1.5,0.0,0.0,0.0,vec3(-49,8,-7),vec3(0.2,1.0,0.1),vec3(0.4,-0.2,0.9),rgb(12,41,60),1,0)
    make(box,5.3,3.9,1.0,0.2,0.7,0.0,vec3(-19,13,12),vec3(0.8,-0.5,0.2),vec3(-0.2,0.1,1.0),rgb(0,0,0),1,2)
    make(cyli,12.2,12.1,9.7,0.7,0.7,0.0,vec3(-17,-40,-25),vec3(-0.0,1.0,0.0),vec3(0.8,0.0,-0.7),rgb(255,212,195),1,0)
    make(cyli,36.8,8.6,36.8,0.3,0.2,0.0,vec3(-20,-50,-18),vec3(-0.1,1.0,0.0),vec3(0.9,0.1,-0.5),rgb(140,123,17),1,0)
    make(cyli,27.4,3.1,26.6,1.0,0.0,0.0,vec3(-20,-42,-18),vec3(-0.1,1.0,-0.0),vec3(1.0,0.1,0.1),rgb(255,255,255),1,0)
    make(tri,31.5,24.3,53.7,0.1,0.0,0.5,vec3(23,-41,33),vec3(0.5,0.5,0.7),vec3(0.8,0.1,-0.7),rgb(145,123,0),1,1)
    make(box,3.8,6.4,1.2,0.2,0.0,0.0,vec3(-45,20,-8),vec3(0.2,-0.1,1.0),vec3(-0.9,0.4,0.3),rgb(0,0,0),1,2)
    make(cyli,25.7,9.1,26.2,0.3,0.6,0.0,vec3(0,43,-25),vec3(0.4,0.9,-0.1),vec3(0.9,-0.4,0.0),rgb(142,134,57),1,0)
    make(elli,11.6,9.1,12.7,0.0,0.0,0.0,vec3(3,48,-26),vec3(0.4,0.9,-0.1),vec3(0.9,-0.4,0.0),rgb(142,134,57),1,0)
    make(cyli,10.6,41.6,11.3,0.2,0.4,0.0,vec3(34,-19,10),vec3(-0.1,1.0,-0.2),vec3(1.0,0.0,-0.1),rgb(142,134,57),1,0)
    make(cyli,8.0,16.1,8.1,0.1,0.1,0.0,vec3(29,32,-3),vec3(-0.2,0.9,-0.3),vec3(1.0,0.1,-0.2),rgb(142,134,57),1,0)
    make(cyli,8.0,12.5,8.1,0.2,0.1,0.0,vec3(19,50,-13),vec3(-0.8,0.5,-0.4),vec3(0.5,0.9,0.1),rgb(142,134,57),1,0)
    make(cyli,8.0,16.1,8.1,0.2,0.0,0.0,vec3(1,49,-25),vec3(-0.6,-0.5,-0.6),vec3(0.6,0.2,-0.8),rgb(142,134,57),1,0)
    make(cyli,10.6,44.9,11.3,0.2,0.4,0.0,vec3(47,-13,7),vec3(-0.1,1.0,-0.2),vec3(1.0,0.0,-0.1),rgb(142,134,57),1,0)
    make(cyli,8.0,16.1,8.1,0.2,0.6,0.0,vec3(39,39,-6),vec3(-0.4,0.9,-0.3),vec3(0.9,0.3,-0.3),rgb(142,134,57),1,0)
    make(cyli,5.3,16.1,5.3,0.2,1.0,0.0,vec3(21,54,-15),vec3(-0.9,0.3,-0.4),vec3(0.5,0.8,-0.4),rgb(142,134,57),1,0)
    make(elli,2.0,10.2,7.5,0.0,0.0,0.0,vec3(11,-1,1),vec3(0.6,0.8,0.1),vec3(0.6,-0.4,-0.7),rgb(255,217,187),1,0)
    make(elli,3.7,3.7,3.7,0.0,0.0,0.0,vec3(11,-19,4),vec3(0.4,0.9,-0.1),vec3(0.6,-0.4,-0.7),rgb(255,255,255),1,0)
    make(cyli,1.0,8.8,1.0,0.0,0.0,0.0,vec3(11,-12,4),vec3(0.0,1.0,-0.1),vec3(0.0,-0.1,-1.0),rgb(255,255,255),1,0)
    make(cyli,34.7,8.6,35.9,0.4,0.1,0.0,vec3(-10,24,-22),vec3(0.4,0.9,-0.1),vec3(0.9,-0.4,0.0),rgb(114,161,255),1,0)
    make(box,5.3,5.9,1.0,0.2,0.0,0.0,vec3(-26,15,10),vec3(0.9,-0.0,0.3),vec3(-0.3,0.1,0.9),rgb(0,0,0),1,2)
    make(box,3.8,8.1,1.0,0.2,0.7,0.0,vec3(-48,20,-17),vec3(0.3,0.2,0.9),vec3(-0.9,0.3,0.2),rgb(0,0,0),1,2)
    make(box,1.1,2.0,1.1,0.2,0.2,0.0,vec3(-40,-21,-7),vec3(0.7,-0.5,0.5),vec3(0.1,-0.5,-0.8),rgb(255,54,58),1,0)
    make(box,1.1,3.2,1.1,0.2,0.2,0.0,vec3(-39,-20,-3),vec3(-0.1,0.6,0.8),vec3(0.8,-0.5,0.4),rgb(255,54,58),1,0)
    make(box,4.6,17.5,6.0,0.2,1.0,0.0,vec3(-36,5,-5),vec3(0.3,0.9,0.0),vec3(0.9,-0.3,0.0),rgb(255,70,70),1,0)
    make(box,5.3,3.9,1.0,0.2,0.7,0.0,vec3(-14,10,13),vec3(0.7,-0.7,0.2),vec3(-0.2,0.1,1.0),rgb(0,0,0),1,2)
    make(elli,8.0,8.0,8.0,0.0,0.0,0.0,vec3(9,55,-18),vec3(-0.6,-0.5,-0.6),vec3(0.6,0.2,-0.8),rgb(142,134,57),1,0)
    make(elli,8.0,8.0,8.0,0.0,0.0,0.0,vec3(27,45,-8),vec3(-0.8,0.5,-0.4),vec3(0.5,0.9,0.1),rgb(142,134,57),1,0)
    make(elli,8.0,19.2,8.0,0.0,0.0,0.0,vec3(20,51,-13),vec3(-0.8,0.5,-0.4),vec3(0.5,0.9,0.1),rgb(142,134,57),1,0)
    make(elli,8.0,15.2,9.5,0.0,0.0,0.0,vec3(5,52,-22),vec3(-0.6,-0.5,-0.6),vec3(0.6,0.2,-0.8),rgb(142,134,57),1,0)
initialize_voxels()
scene.finish()
