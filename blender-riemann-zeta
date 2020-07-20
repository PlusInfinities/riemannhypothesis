import bpy
import numpy as np

"""
https://en.wikipedia.org/wiki/Riemann_zeta_function
Classic definition of the Riemann zeta function, defined for Re(s) > 1
s: complex number input
N: number of terms to sum
"""
def zeta(s, N):
    result = 0
    for n in np.arange(1, N, 1):
        result += 1 / n ** s
    return result


"""
Analytic continuation of the Riemann zeta function using Dirichlet Eta function, defined for Re(s) > 0
s: complex number input
N: number of terms to sum
"""
def zeta_ac(s, N):
    eta = 0
    for n in np.arange(1, N, 1):
        eta += (-1) ** (n - 1) / n ** s
    return 1 / (1 - 2 ** (1 - s)) * eta

 
# useful shortcuts
scene = bpy.context.scene
meshes = bpy.data.meshes
actions = bpy.data.actions
materials = bpy.data.materials

#Remove existing grids
for mesh in meshes:
    bpy.data.meshes.remove(mesh)

#Remove existing actions
for action in actions:
    bpy.data.actions.remove(action)

#Remove existing materials
for material in materials:
    bpy.data.materials.remove(material)

#Add gridded mesh
bpy.ops.mesh.primitive_grid_add(x_subdivisions=49, y_subdivisions=49, size=6, enter_editmode=False, align='WORLD', location=(0, 0, 0))
grid = bpy.context.object

#Set starting frame
scene.frame_set(0)
bpy.ops.anim.insert_keyframe_animall()

#Set final frame
scene.frame_set(24)

#Apply analytically continued zeta function to each valid mesh point
for vertex in meshes['Grid'].vertices:
    #if vertex.co.x > 0 and vertex.co.x != 1: #Weirdly, this doesn't work as well, so include a slight off-set
    if vertex.co.x > .01 and (vertex.co.x < .99 or vertex.co.x > 1.01):
        s = complex(vertex.co.x, vertex.co.y)
        result = zeta_ac(s, 100)
        vertex.co.x = result.real
        vertex.co.y = result.imag
        vertex.co.z = abs(s) #


bpy.ops.anim.insert_keyframe_animall()


#Deletes vertices causing overlap
"""
#Vertices can only be deselected in edit mode
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='DESELECT')

#Vertices can only be selected in object mode
bpy.ops.object.mode_set(mode='OBJECT')
for vertex in grid.data.vertices:
    if vertex.co.z == 0:
       vertex.select = True

#Vertices can only be deleted in edit mode
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.delete(type='VERT')
"""
