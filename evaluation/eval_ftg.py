"""Resumable paired evaluation of Expert/BC/PPO versus tuned FTG."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import tempfile

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import torch
import gym
from scipy.spatial import cKDTree
from f110_gym.envs.base_classes import Integrator
from ftg.controller import FTGConfig, FollowTheGapController
from expert.lattice_planner import create_expert_planner
from expert.utils import downsample_lidar
from f1tenth_sim.utils import load_raceline, racetrack_path, simulation_config
from imitation.model import End2Race

ROOT = Path(__file__).resolve().parent.parent
TRACKS = ["Austin", "Hockenheim", "MoscowRaceway", "Nuerburgring"]
SCALES = [0.5, 0.7, 0.9]
POLICIES = ['expert', 'bc', 'ppo']
_CACHE = {}


class Progress:
    """Closed-polyline projection, independent of simulator start-line toggles."""
    def __init__(self, xy, position):
        self.xy = xy
        self.vec = np.roll(xy, -1, axis=0) - xy
        self.ds = np.linalg.norm(self.vec, axis=1)
        self.s = np.r_[0, np.cumsum(self.ds)]
        self.length = self.s[-1]
        self.tree = cKDTree(xy)
        self.previous = self.project(position)
        self.total = 0.0

    def project(self, position):
        _, ids = self.tree.query(position, k=min(8, len(self.xy)))
        ids = np.unique(np.r_[ids, (ids - 1) % len(self.xy)])
        v = self.vec[ids]
        f = np.clip(np.sum((position - self.xy[ids]) * v, axis=1) / np.maximum(self.ds[ids] ** 2, 1e-12), 0, 1)
        q = self.xy[ids] + f[:, None] * v
        i = np.argmin(np.sum((q - position) ** 2, axis=1))
        return self.s[ids[i]] + f[i] * self.ds[ids[i]]

    def update(self, position):
        cur = self.project(position)
        delta = (cur - self.previous + self.length / 2) % self.length - self.length / 2
        if abs(delta) > 3.0:
            raise RuntimeError("Ambiguous progress projection jump")
        self.total += delta
        self.previous = cur


def pose_at(line, distance):
    """Interpolate a closed line at exact arc distance, including wrapped heading."""
    xy = line[:, :2]
    ds = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    s = np.r_[0., np.cumsum(ds)]
    distance %= s[-1]
    i = min(int(np.searchsorted(s, distance, side='right') - 1), len(line) - 1)
    j = (i + 1) % len(line)
    f = (distance - s[i]) / ds[i]
    yaw_delta = (line[j, 2] - line[i, 2] + np.pi) % (2 * np.pi) - np.pi
    return np.r_[xy[i] + f * (xy[j] - xy[i]), line[i, 2] + f * yaw_delta]


def scenarios(starts=80):
    jobs = []
    for track in TRACKS:
        line = load_raceline(track, 'raceline1.csv')
        length = np.linalg.norm(np.roll(line[:, :2], -1, axis=0) - line[:, :2], axis=1).sum()
        for start in range(starts):
            distance = start * length / starts
            poses = [pose_at(line, distance).tolist(), pose_at(line, distance + 10.).tolist()]
            for scale in SCALES:
                for policy in POLICIES:
                    jobs.append(dict(id=f'{track}_{policy}_{start:02d}_{scale:.1f}', track=track, policy=policy, start=start, start_distance=float(distance), scale=scale, poses=poses, seed=42))
    return jobs


def resources(track, policy):
    torch.set_num_threads(1)
    if _CACHE.get('track') != track:
        if 'env' in _CACHE:
            _CACHE['env'].close()
        _CACHE['env'] = gym.make('f110-v0', map=str(racetrack_path(track, f'{track}_map')), map_ext='.png', num_agents=2, timestep=1/120, integrator=Integrator.RK4, seed=42)
        _CACHE['track'] = track
        _CACHE['line'] = load_raceline(track, 'raceline1.csv')
    key = (track, policy)
    if key not in _CACHE:
        if policy == 'expert':
            planner = create_expert_planner(track, 'raceline1')
            planner.parallel_workers = 1  # Scheduling only; preserve planner candidates and scoring.
            _CACHE[key] = planner
        else:
            model = End2Race()
            model.load_state_dict(torch.load(ROOT / 'checkpoint' / f'{policy}.pt', map_location='cpu', weights_only=True))
            model.eval()
            _CACHE[key] = model
    return _CACHE['env'], _CACHE['line'], _CACHE[key]


def rollout(job, output):
    started = time.monotonic()
    result = dict(job)
    try:
        env, line, ego = resources(job['track'], job['policy'])
        initial_velocities = np.array([3.75, 3.75 * job['scale']])
        # reset() internally takes a physics step. Keep that step stationary,
        # then install initial speeds so t=0 has exactly the requested poses.
        obs, _, _, _ = env.reset(poses=np.asarray(job['poses']), velocities=np.zeros(2))
        env.unwrapped.current_time = 0.0
        for i, speed in enumerate(initial_velocities):
            env.unwrapped.sim.agents[i].state[3] = speed
            obs['linear_vels_x'][i] = float(speed)
        for i in range(2):
            assert np.allclose([obs['poses_x'][i], obs['poses_y'][i]], job['poses'][i][:2], atol=1e-10, rtol=0)
        opponent = FollowTheGapController()
        hidden = None
        previous_speed = 3.75
        if job['policy'] == 'expert':
            ego.best_trajectory = None
        progress = [Progress(line[:, :2], np.asarray(p)[:2]) for p in job['poses']]
        action = np.zeros((2, 2))
        ego_speed_sum = opp_speed_sum = 0.
        ego_collision = opponent_collision = False
        status = 'completed'
        first_pass_time = None
        trajectory = None
        with torch.inference_mode():
            for step in range(1440):
                if job['policy'] == 'expert':
                    if step % 12 == 0:
                        trajectory = ego.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], obs['scans'][0], obs['linear_vels_x'][0])
                    # Preserve the expert collection loop's 120 Hz tracker.
                    action[0] = ego.tracker.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], obs['linear_vels_x'][0], trajectory)
                    action[0, 0] = np.clip(action[0, 0], -0.4189, 0.4189)
                elif step % 3 == 0:
                    scan = downsample_lidar(obs['scans'][0], End2Race.NUM_LIDAR_FEATURES)
                    prediction, hidden = ego(torch.as_tensor(scan, dtype=torch.float32)[None, None], torch.tensor([[[previous_speed]]], dtype=torch.float32), hidden)
                    action[0] = prediction[0, -1].numpy()
                    action[0, 0] = np.clip(action[0, 0], -0.4189, 0.4189)
                    previous_speed = float(obs['linear_vels_x'][0])
                if step % 3 == 0:
                    action[1] = opponent.plan(obs, agent_index=1)
                    action[1, 1] = min(7.5, action[1, 1] * job['scale'])
                if not np.isfinite(action).all():
                    raise ValueError('Nonfinite policy action')
                obs, _, done, _ = env.step(action)
                elapsed = (step + 1) / 120
                ego_speed_sum += float(obs['linear_vels_x'][0])
                opp_speed_sum += float(obs['linear_vels_x'][1])
                ego_collision |= bool(obs['collisions'][0])
                opponent_collision |= bool(obs['collisions'][1])
                for i in range(2):
                    progress[i].update(np.array([obs['poses_x'][i], obs['poses_y'][i]]))
                lead = float(progress[0].total - progress[1].total - 10.)
                if lead > 0.58 and first_pass_time is None:
                    first_pass_time = elapsed
                if ego_collision:
                    status = 'ego_collision'
                    break
                if done:
                    status = 'simulator_done'
                    break
        safe = status == 'completed' and not ego_collision
        result.update(status=status, elapsed_seconds=elapsed, steps=step+1, ego_collision=ego_collision, opponent_collision=opponent_collision, safety=safe, clean_safety=safe and not opponent_collision, overtake=safe and not opponent_collision and lead > 0.58, first_pass_time=first_pass_time, final_lead_m=lead, mean_ego_speed=ego_speed_sum/(step+1), mean_opponent_speed=opp_speed_sum/(step+1), initial_gap_m=10., initial_euclidean_gap_m=float(np.linalg.norm(np.asarray(job['poses'][0])[:2]-np.asarray(job['poses'][1])[:2])), initial_velocities=initial_velocities.tolist())
    except Exception:
        result.update(status='error', error=traceback.format_exc(), safety=False, clean_safety=False, overtake=False)
    result['wall_seconds'] = time.monotonic() - started
    path = Path(output)/'episodes'/f"{job['id']}.json"
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    temporary.replace(path)
    return result


def align_tables(lines):
    i = 0
    while i < len(lines):
        if not lines[i].startswith('|'):
            i += 1
            continue
        start = i
        while i < len(lines) and lines[i].startswith('|'):
            i += 1
        rows = [[c.strip() for c in line.strip('|').split('|')] for line in lines[start:i]]
        widths = [max(len(row[c]) for row in rows) for c in range(len(rows[0]))]
        right = [c.endswith(':') for c in rows[1]]
        for r, row in enumerate(rows):
            cells = []
            for c, value in enumerate(row):
                if r == 1:
                    value = '-' * (widths[c] - int(right[c])) + (':' if right[c] else '')
                else:
                    value = value.rjust(widths[c]) if right[c] else value.ljust(widths[c])
                cells.append(value)
            lines[start+r] = '| ' + ' | '.join(cells) + ' |'
    return lines


def write_report(output, jobs, elapsed, workers=12, report_path=None):
    output = Path(output)
    report_path = report_path or ROOT/'report/FTG_PHASE3_RESULTS.md'
    rows = [json.loads(p.read_text()) for p in sorted((output/'episodes').glob('*.json'))]
    lines = ['# FTG Phase 3: Expert, BC, and PPO', '', f'Completed **{len(rows)}/{len(jobs)} rollouts**. Infrastructure errors: **{sum(r["status"]=="error" for r in rows)}**.', '',
        f'Protocol: four tracks × three ego methods × 80 equally spaced starts × three FTG speed scales (0.5, 0.7, 0.9) = **2,880 rollouts**. {workers} parallel workers; fixed 12 s horizon (1,440 steps at 120 Hz RK4), with early termination on ego collision. Opponent starts exactly 10 m ahead along the closed `raceline1` polyline. All methods share the same start poses and LiDAR seed 42. Initial speeds: ego 3.75 m/s, FTG 3.75 × scale m/s. FTG commands are scaled after planning, capped at 7.5 m/s. The simulator reset step is stationary; initial velocities are installed afterward and the clock is zeroed, preserving the exact initial gap and horizon.', '',
        'Safety = fraction of all requested scenarios completing 12 s without an ego collision. Clean overtake = safety plus no opponent collision and final ego lead greater than one vehicle length (0.58 m). Progress is unwrapped on a shared reference, including the initial 10 m separation. Opponent collisions are latched and reported separately; an overtake of a crashed opponent never counts. Mean speed includes the actual simulated portion of collision episodes and is weighted by simulation time; safe-episode mean is also reported. Errors and unfinished scenarios never count as successes.', '',
        '| Track | Method | Finished / requested | Safety % | Clean overtake % | Ego collision % | Opponent collision % | Mean ego speed m/s | Safe mean m/s |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    summary=[]
    for track in TRACKS:
        for policy in POLICIES:
            group=[r for r in rows if r['track']==track and r['policy']==policy]
            requested=sum(j['track']==track and j['policy']==policy for j in jobs)
            if not requested: continue
            valid=[r for r in group if r['status']!='error']
            safe=[r for r in valid if r['safety']]
            avg=lambda g:sum(r['mean_ego_speed']*r['elapsed_seconds'] for r in g)/max(1e-9,sum(r['elapsed_seconds'] for r in g))
            values=dict(track=track,policy=policy,completed=len(group),requested=requested,safety=100*sum(r['safety'] for r in group)/requested,overtake=100*sum(r['overtake'] for r in group)/requested,ego_collision=100*sum(r.get('ego_collision',False) for r in group)/requested,opponent_collision=100*sum(r.get('opponent_collision',False) for r in group)/requested,mean_speed=avg(valid),safe_mean_speed=avg(safe))
            summary.append(values)
            lines.append(f"| {track} | {policy.upper()} | {len(group)}/{requested} | {values['safety']:.2f} | {values['overtake']:.2f} | {values['ego_collision']:.2f} | {values['opponent_collision']:.2f} | {values['mean_speed']:.3f} | {values['safe_mean_speed']:.3f} |")
    lines += ['', '## Overall results', '', '| Method | Rollouts | Safety % | Clean overtake % | Mean ego speed m/s |', '|---|---:|---:|---:|---:|']
    for policy in POLICIES:
        group = [r for r in rows if r['policy'] == policy]
        n = sum(j['policy'] == policy for j in jobs)
        valid = [r for r in group if r['status'] != 'error']
        speed = sum(r['mean_ego_speed']*r['elapsed_seconds'] for r in valid)/max(1e-9,sum(r['elapsed_seconds'] for r in valid))
        if n:
            lines.append(f"| {policy.upper()} | {len(group)}/{n} | {100*sum(r['safety'] for r in group)/n:.2f} | {100*sum(r['overtake'] for r in group)/n:.2f} | {speed:.3f} |")
    lines += ['', '## Results by opponent speed scale', '', '| Method | Scale | Finished | Safety % | Clean overtake % |', '|---|---:|---:|---:|---:|']
    for policy in POLICIES:
        for scale in SCALES:
            group=[r for r in rows if r['policy']==policy and r['scale']==scale]
            n=sum(j['policy']==policy and j['scale']==scale for j in jobs)
            if n:lines.append(f"| {policy.upper()} | {scale:.1f} | {len(group)}/{n} | {100*sum(r['safety'] for r in group)/n:.2f} | {100*sum(r['overtake'] for r in group)/n:.2f} |")
    lines += ['', 'Expert: original planner at 10 Hz and tracker at 120 Hz; internal candidate workers set to one. BC/PPO: original checkpoints and inference path, recurrent state reset per rollout, 40 Hz control. Tuned FTG: 40 Hz. No policy retraining or parameter tuning during this evaluation.', '', f'Runner wall time this invocation: {elapsed/60:.1f} minutes.', '']
    (output/'summary.json').write_text(json.dumps(dict(completed=len(rows),requested=len(jobs),errors=sum(r['status']=='error' for r in rows),summary=summary),indent=2)+'\n')
    lines = align_tables(lines)
    report_path.write_text('\n'.join(lines))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--output',type=Path,help='Optional persistent work directory for resuming; otherwise intermediate files are temporary.')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    temporary = tempfile.TemporaryDirectory(prefix='ftg_phase3_') if args.output is None else None
    args.output = Path(temporary.name) if temporary else args.output
    report_path = (args.output/'REPORT.md') if args.smoke else ROOT/'report/FTG_PHASE3_RESULTS.md'
    if args.smoke and temporary:
        report_path = ROOT/'report/FTG_SMOKE_RESULTS.md'
    jobs=scenarios()
    if args.smoke:jobs=[j for j in jobs if j['start']==0 and j['scale']==0.7]
    sim=simulation_config()
    assert sim.frequency_hz==120 and sim.control_frequency_hz==40
    for sub in ['episodes']: (args.output/sub).mkdir(parents=True,exist_ok=True)
    source=[ROOT/'evaluation/eval_ftg.py',ROOT/'ftg/controller.py',ROOT/'expert/lattice_planner.py',ROOT/'expert/controllers.py',ROOT/'expert/config.yaml',ROOT/'imitation/model.py',ROOT/'checkpoint/bc.pt',ROOT/'checkpoint/ppo.pt',ROOT/'f1tenth_sim/config.yaml',ROOT/'expert/utils.py',ROOT/'f1tenth_sim/utils.py',ROOT/'f1tenth_sim/f1tenth_racetracks/config.yaml']
    source.extend((ROOT/'f1tenth_sim/f1tenth_gym/gym/f110_gym/envs').glob('*.py'))
    for track in TRACKS:
        source.extend([racetrack_path(track,'raceline1.csv'),racetrack_path(track,f'{track}_map.png'),racetrack_path(track,f'{track}_map.yaml')])
    protocol=dict(workers=args.workers,seconds=12,gap_m=10,scales=SCALES,starts=80,seed=42,ftg=asdict(FTGConfig()),jobs=jobs,sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source})
    protocol_path=args.output/'protocol.json'
    if protocol_path.exists():
        old=json.loads(protocol_path.read_text())
        if old!=protocol:raise RuntimeError('Protocol/source mismatch: use a fresh output directory')
    else:protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    pending=[j for j in jobs if not (args.output/'episodes'/f"{j['id']}.json").exists()]
    started=time.monotonic()
    done=len(jobs)-len(pending)
    write_report(args.output,jobs,0.,args.workers,report_path)
    print(f'Starting {len(pending)} pending / {len(jobs)} total; workers={args.workers}',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(rollout,j,args.output) for j in pending]
        for future in as_completed(futures):
            r=future.result();done+=1
            if r['status']=='error':print(r['error'],flush=True)
            if done%40==0 or done==len(jobs) or args.smoke:
                elapsed=time.monotonic()-started
                rows=write_report(args.output,jobs,elapsed,args.workers,report_path)
                rate=(done-(len(jobs)-len(pending)))/max(elapsed,1e-6)
                print(f"{done}/{len(jobs)} | {r['track']} {r['policy']} | errors={sum(x['status']=='error' for x in rows)} | elapsed={elapsed/60:.1f}m | ETA={(len(jobs)-done)/max(rate,1e-9)/60:.1f}m",flush=True)
    rows=write_report(args.output,jobs,time.monotonic()-started,args.workers,report_path)
    if temporary:
        temporary.cleanup()
    return int(any(r['status']=='error' for r in rows))

if __name__=='__main__':raise SystemExit(main())
