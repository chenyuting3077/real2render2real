"""Check the OpenArm adapter against the actual upstream dual-arm target method."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

import real2render2real.utils.transforms as tf
from real2render2real.isaaclab_viser.openarm_simulators.cardboard_handoff import (
    CardboardHandoffConfig, cardboard_target, compatible_grasp_indices, hand_approach_offset, reference_grasp_frames,
)
from real2render2real.isaaclab_viser.openarm_simulators.video_bimanual import VideoContactTimeline, pose_matrix


class CardboardHandoffTests(unittest.TestCase):
    def test_targets_match_actual_upstream_method_for_both_arms_and_all_offsets(self):
        path = Path(__file__).resolve().parents[1]/"real2render2real/isaaclab_viser/yumi_simulators/yumi_cardboard_lift.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CardboardLift")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_calculate_target_poses")
        namespace = dict(torch=torch, onp=np, tf=tf)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        origins = torch.tensor([[0., 0., 0.], [2., 0., 0.]])
        states = torch.tensor([[.4, .1, .5, .70710678, 0, 0, .70710678],
                               [2.3, -.2, .4, .9238795, .3826834, 0, 0]])
        grasps = [dict(position=[.01, -.03, -.04], orientation=[1.,0,0,0]),
                  dict(position=[-.02, .03, .05], orientation=[.70710678,.70710678,0,0])]
        cfg = SimpleNamespace(setup_phase_steps=13,grasp_phase_steps=80,resampled=False,release_phase_steps=lambda _:600)
        sim = SimpleNamespace(scene=SimpleNamespace(num_envs=2,env_origins=origins),
                              grasp_data=[grasps], grasp_perturb=None,
                              left_hand_envs=torch.zeros(2,dtype=torch.bool),
                              state_machine=SimpleNamespace(config=cfg), _update_ee_poses=lambda:None)
        sim._get_ee_poses = lambda side:(torch.zeros((2,3)),torch.tensor([[1.,0,0,0]]).repeat(2,1))
        obj = SimpleNamespace(data=SimpleNamespace(root_state_w=states))
        for offset in (-.105,-.005,-.1):
            sim.state_machine.ee_goal_offset = [0,0,offset]
            upstream = namespace["_calculate_target_poses"](sim,100,obj)
            for env in range(2):
                p=states[env].numpy().copy();p[:3]-=origins[env].numpy()
                for i,g in enumerate((grasps[1],grasps[0])):  # official order: left, right
                    adapted=cardboard_target(pose_matrix(np.r_[p[3:],p[:3]]),g,offset)
                    np.testing.assert_allclose(adapted[:3],upstream[env,i,:3],atol=2e-7)
                    q=upstream[env,i,3:]
                    self.assertAlmostEqual(abs(float(adapted[3:]@q)),1,places=6)

    def test_offset_is_in_gripper_frame_not_world_vertical(self):
        grasp=dict(position=[0,0,0],orientation=[.70710678,0,.70710678,0])
        target=cardboard_target(np.eye(4),grasp,-.1)
        np.testing.assert_allclose(target[:3],[-.1,0,0],atol=1e-7)

    def test_video_timing_keeps_right_then_overlap_then_left(self):
        cfg=CardboardHandoffConfig()
        data={"hands":{s:{"contact":c} for s,c in (
            ("left",[False]*3+[True]*4+[False]*3),
            ("right",[True]*5+[False]*5))}}
        timeline=VideoContactTimeline(data,cfg,raw_start=0,upsample=6)
        for frame,expected in [(0,(False,True)),(3,(True,True)),(5,(True,False)),(7,(False,False))]:
            count=cfg.grasp_phase_steps+frame*6
            timeline.update(count)
            self.assertEqual((timeline.l_gripper_closed,timeline.r_gripper_closed),expected)
        count=cfg.grasp_phase_steps+5*6
        self.assertEqual(hand_approach_offset(timeline,"left",count),(-.005,None))
        offset,anchor=hand_approach_offset(timeline,"right",count)
        self.assertEqual(offset,-.005)
        self.assertEqual(anchor,4*6)
        final=count+cfg.release_offset_steps
        self.assertEqual(hand_approach_offset(timeline,"right",final),(-.1,4*6))

    def test_rejects_tiny_surface_grasp_but_keeps_body_grasp(self):
        vertices=np.array([[x,y,z] for x in (-.035,.035) for y in (-.035,.035) for z in (-.1,.1)])
        body=dict(position=[0,0,0],orientation=[1,0,0,0],width=.034)
        thin=dict(position=[0,0,0],orientation=[1,0,0,0],width=.000047)
        vertical=dict(position=[0,0,0],orientation=[.70710678,.70710678,0,0],width=.035)
        outside=dict(position=[.2,0,0],orientation=[1,0,0,0],width=.035)
        self.assertEqual(compatible_grasp_indices(vertices,[thin,body,vertical,outside]),[1])

    def test_reference_roll_preserves_video_axis_and_uses_each_robot_home(self):
        axis_rotation=Rotation.from_euler('xyz',[.1,.2,.3])
        g=dict(position=[.01,.02,-.06],orientation=axis_rotation.as_quat()[[3,0,1,2]].tolist(),width=.03)
        obj=Rotation.from_euler('z',2.99).as_matrix()
        home=np.array([[0,0,1],[0,-1,0],[1,0,0]],dtype=float)
        frames=reference_grasp_frames([g,g],[0,1],obj,home)
        self.assertEqual(len(frames),1)
        f=frames[0]
        np.testing.assert_array_equal(f['position'],g['position'])
        r=Rotation.from_quat(np.asarray(f['orientation'])[[1,2,3,0]]).as_matrix()
        self.assertAlmostEqual(abs(float(r[:,1]@axis_rotation.as_matrix()[:,1])),1,places=6)
        self.assertGreater((obj@r)[:,0]@home[:,0],.9)
        self.assertGreater((obj@r)[:,2]@home[:,2],0)
        self.assertEqual(f['grasp_frame'],'urdf_home_reference')

    def test_released_right_target_stays_at_video_anchor_while_left_continues(self):
        path=Path(__file__).resolve().parents[1]/'real2render2real/isaaclab_viser/openarm_simulators/openarm_pringles_bimanual.py'
        cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef))
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_calculate_target_poses')
        namespace=dict(np=np,pose_matrix=pose_matrix,SIDES=('left','right'),
                       hand_approach_offset=hand_approach_offset,cardboard_target=cardboard_target)
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
        cfg=CardboardHandoffConfig()
        timeline=VideoContactTimeline({'hands':{'right':{'contact':[True]*3+[False]*3},
                                              'left':{'contact':[False]*2+[True]*4}}},cfg,raw_start=0,upsample=2)
        state=torch.tensor([[1.,0,.5,1,0,0,0]])
        deltas=torch.zeros((11,1,7));deltas[:,:,0]=1;deltas[:,0,4]=torch.arange(11)*.1
        grasp=dict(position=[0,0,0],orientation=[1,0,0,0])
        sim=SimpleNamespace(scene=SimpleNamespace(num_envs=1,env_origins=torch.zeros((1,3))),
                            observed_approach=SimpleNamespace(paths={}), home_ee_poses={s:np.array([[0,0,.5,1,0,0,0]]) for s in ('left','right')},
                            replay_feasible=np.array([True]),state_machine=timeline,
                            parts_init_state={'pringles':torch.tensor([[.3,0,.5,1,0,0,0]])},
                            part_deltas=deltas,_scale=2.,_grasp_for=lambda s,e:grasp)
        obj=SimpleNamespace(data=SimpleNamespace(root_state_w=state))
        count=cfg.grasp_phase_steps+6
        first=namespace['_calculate_target_poses'](sim,count,obj).copy()
        np.testing.assert_allclose(first[0,1,:3],[.5,0,.495],atol=1e-7)
        state[:,:3]+=torch.tensor([.1,-.2,.3])
        second=namespace['_calculate_target_poses'](sim,count,obj)
        np.testing.assert_allclose(second[0,1],first[0,1])
        np.testing.assert_allclose(second[0,0,:3]-first[0,0,:3],[.1,-.2,.3],atol=1e-7)


if __name__ == "__main__":
    unittest.main()
