# data/collect_cam.py
# Collecte AVEC CAMERA embarquee (Utah, terre), pour donner au JEPA des
# observations riches : c'est la ou l'espace latent est cense apporter quelque chose.
#
# Par rapport a collect_slide.py :
#   - camera couleur 64 x 64 sur le toit, regardant devant, lue toutes les 5 pas (10 images/s) ;
#   - images sauvegardees par run dans data/frames_cam/run_XXXX.npz (frames, steps) ;
#   - dynamique dans data/raw_runs_cam.csv (meme format que raw_runs_slide.csv) ;
#   - melange conduite normale / glisse provoquee (50 / 50), nouvelle graine ;
#   - BeamNG est RELANCE completement tous les 80 runs (les nuits 3 et 4, l'IA
#     cessait de conduire apres ~88 runs, meme apres rechargement du scenario) ;
#   - mode --test : 2 runs courts + verification automatique des images + apercu PNG.
#     Code de sortie 0 si les images sont bonnes, 2 sinon.
#
# API camera verifiee dans le code source de BeamNGpy 1.36 :
#   Camera(name, bng, vehicle, requested_update_time, pos, dir, resolution, ...),
#   dir=(0, -1, 0) = vers l'avant (repere vehicule), poll()["colour"] = image PIL.

import os
import sys
import csv
import math
import random
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(HERE))
sys.path.append(HERE)
import config as C
from collect_beamng import HOME, VEHICLE_MODEL, STEPS_PER_SECOND, body_frame, wrap_angle
from collect_slide import SPAWNS, LEVEL, num

OUT_CSV = os.path.join(C.DATA_DIR, "raw_runs_cam.csv")
FRAMES_DIR = os.path.join(C.DATA_DIR, "frames_cam")
PREVIEW = os.path.join(C.OUT_DIR, "apercu_camera.png")

N_RUNS = 450
RUN_SECONDS = 20
RELAUNCH_EVERY = 80
CAM_EVERY = 5                 # 1 image tous les 5 pas = 10 images/s
CAM_RES = (64, 64)
CAM_POS = (0.0, 0.0, 1.8)     # au-dessus du centre de la voiture (repere vehicule, z vers le haut)
CAM_DIR = (0.0, -1.0, -0.15)  # vers l'avant (-y), legerement vers le bas
WARMUP_SECONDS = 5
STUCK_SPEED, STUCK_SECONDS = 1.0, 2
CRASH_TRIM_SECONDS, CRASH_ACC = 2, 40.0
SET_SPEED_PROB = 0.5
SET_SPEED_RANGE, LIMIT_SPEED_RANGE = (12, 24), (10, 25)
AGGRESSION_RANGE = (0.5, 1.0)
MAX_DEAD_IN_A_ROW = 3

EXTRA_ELECTRICS = ["rpm", "gear_index", "engine_load", "accZSmooth", "abs_active", "esc_active", "tcs_active"]
HEADER = (["run_id", "t"] + C.STATE_COLS + C.ACTION_COLS +
          ["throttle", "brake", "acc_x", "acc_y", "wheelspeed", "aggression", "speed_target", "speed_mode",
           "spawn_id", "x", "y", "z"] + EXTRA_ELECTRICS)


class Sim:
    """BeamNG + scenario + voiture + camera, relancable entierement."""
    def __init__(self):
        self.bng = self.veh = self.cam = None

    def start(self):
        from beamngpy import BeamNGpy, Scenario, Vehicle
        from beamngpy.sensors import Electrics, Camera
        self.bng = BeamNGpy("localhost", 25252, home=HOME)
        self.bng.open()
        scenario = Scenario(LEVEL, "rally_jepa_cam")
        self.veh = Vehicle("ego", model=VEHICLE_MODEL, license="JEPA")
        self.veh.sensors.attach("electrics", Electrics())
        x, y, z = SPAWNS[0]
        scenario.add_vehicle(self.veh, pos=(x, y, z + 0.5), rot_quat=(0, 0, 0, 1))
        scenario.make(self.bng)
        self.bng.settings.set_deterministic(STEPS_PER_SECOND)
        self.bng.scenario.load(scenario)
        self.bng.scenario.start()
        self.cam = Camera("cam", self.bng, self.veh, requested_update_time=CAM_EVERY / STEPS_PER_SECOND,
                          pos=CAM_POS, dir=CAM_DIR, resolution=CAM_RES, field_of_view_y=70,
                          is_render_colours=True, is_render_annotations=False, is_render_depth=False)

    def stop(self):
        try:
            if self.cam is not None:
                self.cam.remove()
        except Exception:
            pass
        try:
            if self.bng is not None:
                self.bng.close()
        except Exception:
            pass
        self.bng = self.veh = self.cam = None
        time.sleep(10)

    def restart(self):
        self.stop()
        self.start()


def grab(cam):
    """Image courante (64, 64, 3) uint8, ou None."""
    try:
        img = cam.poll().get("colour")
    except Exception:
        return None
    if img is None:
        return None
    a = np.asarray(img, dtype=np.uint8)
    return a[:, :, :3] if a.ndim == 3 else None


def record_run(sim, run_id, spawn_id, n_steps, rng):
    bng, veh = sim.bng, sim.veh
    aggression = round(rng.uniform(*AGGRESSION_RANGE), 2)
    if rng.random() < SET_SPEED_PROB:
        mode, target = "set", round(rng.uniform(*SET_SPEED_RANGE), 1)
    else:
        mode, target = "limit", round(rng.uniform(*LIMIT_SPEED_RANGE), 1)
    veh.ai.set_mode("random")
    veh.ai.set_aggression(aggression)
    veh.ai.set_speed(target, mode=mode)

    rows, frames, fsteps = [], [], []
    prev_heading = prev_time = prev_vel = None
    stuck = 0
    warmup = WARMUP_SECONDS * STEPS_PER_SECOND
    stuck_limit = STUCK_SECONDS * STEPS_PER_SECOND
    trim = CRASH_TRIM_SECONDS * STEPS_PER_SECOND
    reason = None
    for t in range(n_steps):
        bng.control.step(1)
        veh.sensors.poll()
        st, el = veh.state, veh.sensors["electrics"]
        vx, vy, heading = body_frame(st)
        sim_time = st.get("time")
        dt = C.DT
        if prev_time is not None and sim_time is not None and sim_time > prev_time:
            dt = sim_time - prev_time
        r = 0.0 if prev_heading is None else wrap_angle(heading - prev_heading) / dt
        vel = st["vel"]
        acc_h = 0.0 if prev_vel is None else math.hypot(vel[0] - prev_vel[0], vel[1] - prev_vel[1]) / dt
        prev_heading, prev_time, prev_vel = heading, sim_time, vel
        thr, brk = num(el.get("throttle", 0.0)), num(el.get("brake", 0.0))
        pos = st["pos"]
        rows.append([run_id, t, vx, vy, r, num(el.get("steering", 0.0)), thr - brk, thr, brk,
                     num(el.get("accXSmooth")), num(el.get("accYSmooth")), num(el.get("wheelspeed")),
                     aggression, target, mode, spawn_id, pos[0], pos[1], pos[2]] +
                    [num(el.get(k)) for k in EXTRA_ELECTRICS])
        if t % CAM_EVERY == 0:
            img = grab(sim.cam)
            if img is not None:
                frames.append(img); fsteps.append(t)
        if t > 1 and acc_h > CRASH_ACC:
            rows = rows[:max(0, len(rows) - trim)]
            reason = f"choc ({acc_h:.0f} m/s2) a t={t / STEPS_PER_SECOND:.1f}s"
            break
        stuck = stuck + 1 if (t > warmup and abs(vx) < STUCK_SPEED) else 0
        if stuck >= stuck_limit:
            rows = rows[:max(0, len(rows) - stuck_limit - trim)]
            reason = f"voiture bloquee a t={t / STEPS_PER_SECOND:.1f}s"
            break
    keep = len(rows)
    sel = [k for k, s in enumerate(fsteps) if s < keep]
    frames = np.array([frames[k] for k in sel], dtype=np.uint8).reshape(-1, CAM_RES[1], CAM_RES[0], 3)
    fsteps = np.array([fsteps[k] for k in sel], dtype=np.int32)
    return rows, frames, fsteps, aggression, mode, target, reason


def save_preview(all_frames):
    """Planche de 4 x 6 images pour verifier que la camera regarde devant."""
    from PIL import Image
    imgs = [f for fr in all_frames for f in fr[::max(1, len(fr) // 12)]][:24]
    if not imgs:
        return
    h, w = CAM_RES[1], CAM_RES[0]
    sheet = np.zeros((4 * h, 6 * w, 3), np.uint8)
    for k, im in enumerate(imgs):
        sheet[(k // 6) * h:(k // 6 + 1) * h, (k % 6) * w:(k % 6 + 1) * w] = im
    os.makedirs(C.OUT_DIR, exist_ok=True)
    Image.fromarray(sheet).resize((6 * w * 3, 4 * h * 3), Image.NEAREST).save(PREVIEW)
    print(f"[apercu] {PREVIEW}", flush=True)


def collect(n_runs, run_seconds, test=False):
    rng = random.Random(C.SEED + 3)                  # nouvelle graine : pas de runs jumeaux
    spawns = [(x, y, z + 0.5) for x, y, z in SPAWNS]
    os.makedirs(FRAMES_DIR, exist_ok=True)
    sim = Sim(); sim.start()
    total, dead, since_launch, test_frames = 0, 0, 0, []
    expected = run_seconds * STEPS_PER_SECOND // CAM_EVERY
    n_frames_ok = n_frames_exp = 0
    try:
        with open(OUT_CSV if not test else OUT_CSV.replace(".csv", "_test.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(HEADER)
            for run_id in range(n_runs):
                if since_launch >= RELAUNCH_EVERY:
                    print(f"  [relance complete de BeamNG apres {since_launch} runs]", flush=True)
                    sim.restart(); since_launch = 0
                spawn_id = rng.randrange(len(spawns))
                sim.veh.teleport(spawns[spawn_id], reset=True)
                t0 = time.time()
                try:
                    rows, frames, fsteps, agg, mode, target, reason = record_run(
                        sim, run_id, spawn_id, run_seconds * STEPS_PER_SECOND, rng)
                except Exception as e:
                    print(f"  run {run_id} : erreur ({e}), relance de BeamNG", flush=True)
                    sim.restart(); since_launch = 0
                    continue
                since_launch += 1
                ai_dead = (reason is not None and reason.startswith("voiture bloquee")
                           and sum(r[7] for r in rows) <= 0.01 * max(1, len(rows)))
                if ai_dead:
                    dead += 1
                    print(f"  run {run_id} : IA inactive ({dead}/{MAX_DEAD_IN_A_ROW}), relance complete de BeamNG", flush=True)
                    if dead >= MAX_DEAD_IN_A_ROW:
                        print("  IA toujours inactive : arret de la collecte.", flush=True)
                        break
                    sim.restart(); since_launch = 0
                    continue
                dead = 0
                w.writerows(rows); f.flush(); total += len(rows)
                if not test:
                    np.savez_compressed(os.path.join(FRAMES_DIR, f"run_{run_id:04d}.npz"), frames=frames, steps=fsteps)
                else:
                    test_frames.append(frames)
                    n_frames_ok += len(frames)
                    n_frames_exp += len(rows) // CAM_EVERY
                msg = (f"  run {run_id} : depart {spawn_id}, {len(rows)} pas, {len(frames)} images, {mode} {target} m/s, "
                       f"agressivite {agg}, {time.time() - t0:.0f} s reelles")
                print(msg + (f"  [coupe : {reason}]" if reason else ""), flush=True)
    finally:
        sim.stop()
    print(f"[collect] {total} lignes", flush=True)

    if test:
        save_preview(test_frames)
        allf = np.concatenate(test_frames) if test_frames and sum(len(x) for x in test_frames) else np.zeros((0,))
        ratio = n_frames_ok / max(1, n_frames_exp)
        spread = float(allf.std()) if allf.size else 0.0
        print(f"[test] images recues : {n_frames_ok}/{n_frames_exp} ({ratio:.0%}), dispersion des pixels : {spread:.1f}", flush=True)
        ok = ratio >= 0.8 and spread > 5.0
        print("[test] CAMERA OK" if ok else "[test] CAMERA EN ECHEC : la collecte complete ne sera pas lancee", flush=True)
        sys.exit(0 if ok else 2)


if __name__ == "__main__":
    if "--test" in sys.argv:
        collect(n_runs=2, run_seconds=10, test=True)
    else:
        collect(n_runs=N_RUNS, run_seconds=RUN_SECONDS)
