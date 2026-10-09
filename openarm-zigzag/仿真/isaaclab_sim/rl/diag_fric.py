import argparse, sys, os
parser = argparse.ArgumentParser()
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); args.headless = True
app = AppLauncher(args); simulation_app = app.app
sys.path.insert(0, '/home/yang/桌面/openarm实验/仿真/isaaclab_sim/rl')
import numpy as np, torch
from openarm_zigzag.zigzag_env import OpenArmZigzagEnv, OpenArmZigzagEnvCfg

cfg = OpenArmZigzagEnvCfg(); cfg.scene.num_envs = 8; cfg.debug_terms = True
env = OpenArmZigzagEnv(cfg); u = env.unwrapped
obs, _ = env.reset()
for i in range(300):
    a = torch.tanh(0.3*torch.randn(8,7, device=env.device))
    obs, rew, term, trunc, info = env.step(a)
    if i in (99, 299):
        d = u._dbg
        print("step %d rew %.2f | track %.2f acc %.3f vel_lim %.3f limit %.3f"
              % (i, float(rew.mean()), d["track"], d["acc"], d["vel_lim"], d["limit"]), flush=True)
        print("        fail %.2f fric_b %.3f fric_c %.3f"
              % (d["fail"], float(u._arm_act._fric_b.mean()),
                 float(u._arm_act._fric_c.mean())), flush=True)
simulation_app.close(); os._exit(0)
