"""Passive actual USD/PhysX collider audit; no simulation or control writes."""
import math
import numpy as np

TOLERANCES = {'position_m': 1.e-4, 'dimension_m': 1.e-5, 'axis_absolute': 1.e-5,
              'pose_roundtrip_m': 1.e-4, 'pair_contact_force_N': 1.e-6}


def read_box(stage, body, box, origins, index, transform_reader=None):
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    from camra_navigation.envs.clean_v3 import rigid_body_transform
    pose = (transform_reader or rigid_body_transform)(body, index)
    path = '/World/envs/env_'+str(index)+'/'+body.cfg.prim_path.rsplit('/', 1)[-1]
    prim = stage.GetPrimAtPath(path)
    cubes = [p for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Cube)
             and p.HasAPI(UsdPhysics.CollisionAPI)
             and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
    if len(cubes) != 1:
        raise ValueError('exactly one actual enabled Cube is required: '+path)
    cube = cubes[0]
    size = float(UsdGeom.Cube(cube).GetSizeAttr().Get())
    matrix = UsdGeom.Xformable(cube).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    center = np.asarray(matrix.Transform(Gf.Vec3d(0., 0., 0.)), dtype=float)
    axes = np.array([matrix.TransformDir(Gf.Vec3d(*v)) for v in np.eye(3)], dtype=float)
    lengths = np.linalg.norm(axes, axis=1)
    dims = size*lengths
    expected = np.array([box.center_x+origins[index,0], box.center_y+origins[index,1],
                         box.z_center+origins[index,2]])
    desired = np.array([box.size_x,box.size_y,box.height])
    c,s = math.cos(box.yaw),math.sin(box.yaw)
    basis = np.array([[c,s,0.],[-s,c,0.],[0.,0.,1.]])
    errors = {'position_m': float(np.max(np.abs(center-expected))),
              'physx_position_m':float(np.max(np.abs(pose[:3]-expected))),
              'dimension_m':float(np.max(np.abs(dims-desired))),
              'axis_absolute':float(np.max(np.abs(axes/lengths[:,None]-basis)))}
    if (max(errors['position_m'],errors['physx_position_m'])>TOLERANCES['position_m']
        or errors['dimension_m']>TOLERANCES['dimension_m']
        or errors['axis_absolute']>TOLERANCES['axis_absolute']):
        raise ValueError('actual composed collider geometry mismatch: '+str(errors))
    return {'env_id':index,'body_path':path,'cube_path':str(cube.GetPath()),
            'cube_size_attribute':size,'composed_world_matrix':[[float(v) for v in row] for row in matrix],
            'actual_world_center':center.tolist(),'actual_world_axis_vectors':axes.tolist(),
            'actual_dimensions_m':dims.tolist(),'actual_physx_xyz_xyzw':pose.tolist(),
            'expected_box':box.as_dict(),'actual_env_origin':origins[index].tolist(),
            'errors':errors,'enabled':True}


def read_ground(stage, path):
    from pxr import Usd, UsdPhysics, UsdShade
    rows=[]
    for p in Usd.PrimRange(stage.GetPrimAtPath(path)):
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            continue
        material,_=UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial(materialPurpose='physics')
        if material:
            api=UsdPhysics.MaterialAPI(material.GetPrim())
            values=[api.GetStaticFrictionAttr().Get(),api.GetDynamicFrictionAttr().Get(),api.GetRestitutionAttr().Get()]
            if values != [1.,1.,0.]:
                raise ValueError('actual ground is not fixed clean material')
            rows.append({'collision_prim':str(p.GetPath()),'material':str(material.GetPath()),'values':values})
    if not rows:
        raise ValueError('no actual ground material binding')
    return rows
