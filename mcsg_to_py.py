import json
import numpy as np
import argparse

def parse_mcsg_to_json(path):
    with open(path) as f:
        output = []
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            if line.endswith(':\n') or line.endswith('{\n') or line.endswith('[\n'):
                output.append(line)
                continue
            if i != len(lines) - 1 and "}" not in lines[i+1] and "]" not in lines[i+1]:
                newline = line.replace('\n', ",\n")
                output.append(newline)
                continue
            output.append(line)
        output.insert(0, "{\n")
        output.append("}\n")
        data = json.loads("".join(output))
        return data

def get_rot(json):
    rotation_matrix = json['r']
    r = rotation_matrix.split(' ')
    rot = np.zeros((9,1), np.float32)
    for i in range(9):
        rot[i] = float(r[i])
    return rot.reshape([3,3]).transpose()

def get_vec(line):
    numbers = line.split(' ')
    ret = np.zeros((3), np.float32)
    for i in range(3):
        ret[i] = float(numbers[i])
    return ret

def make_code(type, size, extra_param, pos, forward, up, color, mode):
    return "    make({0},{1:.1f},{2:.1f},{3:.1f},{4:.1f},{5:.1f},{6:.1f},vec3({7:.0f},{8:.0f},{9:.0f}),vec3({10:.1f},{11:.1f},{12:.1f}),vec3({13:.1f},{14:.1f},{15:.1f}),rgb({16:.0f},{17:.0f},{18:.0f}),1,{19:d})\n".format(\
        type, size[1], size[2], size[0], extra_param[0], extra_param[1], extra_param[2], round(pos[1]), round(pos[2]-64), round(pos[0]), forward[1], forward[2], forward[0], up[1], up[2], up[0], color[0], color[1], color[2], mode)

PRE_CODE = """from scene import Scene
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
"""

POST_CODE = """initialize_voxels()
scene.finish()
"""

def main(mcsg_filepath, out_filepath):
    models = parse_mcsg_to_json(mcsg_filepath)
    outlines = []
    outlines.append(PRE_CODE)
    for model in models['csg'][0]:
        mode = 0
        if "mode" in model:
            mode = 1 if model["mode"] == "sub" else 2
        rotation_matrix = get_rot(model)
        forward = np.matmul(rotation_matrix, np.array([0,0,1]))
        up = np.matmul(rotation_matrix, np.array([0,1,0]))
        if model['type'] == 'sphere':
            code = make_code("elli", get_vec(model["s"]), [0,0,0], get_vec(model["t"]), forward, up, get_vec(model["rgb"]), mode)
        if model['type'] == 'cube':
            rd = float(model['round%']) if "round%" in model else 0
            cone = float(model['cone%']) if "cone%" in model else 0
            bevel = float(model['bevel%']) if "bevel%" in model else 0
            code = make_code("box", get_vec(model["s"]), [rd, cone, bevel], get_vec(model["t"]), forward, up, get_vec(model["rgb"]), mode)
        if model['type'] == 'cylinder':
            rd = float(model['round%']) if "round%" in model else 0
            cone = float(model['cone%']) if "cone%" in model else 0
            hole = float(model['hole%']) if "hole%" in model else 0
            code = make_code("cyli", get_vec(model["s"]), [rd, cone, hole], get_vec(model["t"]), forward, up, get_vec(model["rgb"]), mode)
        if model['type'] == 'triangle':
            rd = float(model['round%']) if "round%" in model else 0
            cone = float(model['cone%']) if "cone%" in model else 0
            top_v = 0.5 - 0.5 * float(model['top_v%']) if "top_v%" in model else 0.5
            code = make_code("tri", get_vec(model["s"]), [rd, cone, top_v], get_vec(model["t"]), forward, up, get_vec(model["rgb"]), mode)
        print(code)
        outlines.append(code)
    outlines.append(POST_CODE)
    with open(out_filepath, "w") as f:
        f.writelines(outlines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_mcsg_filepath", default="girl.mcsg", type=str)
    parser.add_argument("-o", "--out_code_filepath", default="main.py", type=str)
    args = parser.parse_args()
    main(args.in_mcsg_filepath, args.out_code_filepath)