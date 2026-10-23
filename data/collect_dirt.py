# data/collect_dirt.py
# Collecte sur routes en terre (carte Utah), pour avoir enfin de la glisse.
# Reutilise les fonctions de collect_beamng.py. Differences :
#   - plusieurs points de depart (SPAWNS), un tire au hasard a chaque run ;
#   - detection des chocs par pic d'acceleration, en plus de "voiture bloquee" ;
#   - la position (x, y, z) est enregistree, pour verifier ensuite ou l'IA a roule ;
#   - ecrit dans data/raw_runs_dirt.csv : ton CSV asphalte n'est PAS ecrase.
#
# Test rapide (2 runs de 20 s) :  python data/collect_dirt.py --test
# Collecte complete            :  python data/collect_dirt.py

import os
import sys
import csv
import math
import random
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(HERE))
sys.path.append(HERE)
import config as C
from collect_beamng import HOME, VEHICLE_MODEL, STEPS_PER_SECOND, body_frame, wrap_angle

LEVEL = "utah"
OUT_CSV = os.path.join(C.DATA_DIR, "raw_runs_dirt.csv")

# --- A REMPLIR : positions (x, y, z) relevees sur des pistes en terre ---
SPAWNS = [
    (541.12, -517.64, 133.95),
    (489.36, -699.02, 124.39),
    (-338.98, -509.37, 131.32),
    (38.80, -96.32, 146.59),
    (197.43, 236.09, 184.47),
]

N_RUNS = 40
RUN_SECONDS = 60
WARMUP_SECONDS = 5
STUCK_SPEED = 1.0
STUCK_SECONDS = 2
CRASH_TRIM_SECONDS = 2
CRASH_ACC = 40.0          # m/s2 horizontal (~4 g) : au-dessus, c'est un choc, pas de la conduite
SPEED_RANGE = (8, 22)     # m/s, soit environ 30 a 80 km/h : plus lent que sur asphalte

EXTRA_COLS = ["throttle", "brake", "acc_x", "acc_y", "wheelspeed",
              "aggression", "speed_limit", "spawn_id", "x", "y", "z"]
HEADER = ["run_id", "t"] + C.STATE_COLS + C.ACTION_COLS + EXTRA_COLS


def record_run(bng, veh, run_id, spawn_id, n_steps, rng):
    aggression = round(rng.uniform(0.3, 1.0), 2)
    speed_limit = round(rng.uniform(*SPEED_RANGE), 1)
    veh.ai.set_mode("random")
    veh.ai.set_aggression(aggression)
    try:
        veh.ai.set_speed(speed_limit, mode="limit")
    except Exception as e:
        print("  (limite de vitesse non appliquee :", e, ")")

    rows = []
    prev_heading = prev_time = prev_vel = None
    stuck_steps = 0
    warmup = WARMUP_SECONDS * STEPS_PER_SECOND
    stuck_limit = STUCK_SECONDS * STEPS_PER_SECOND
    trim_steps = CRASH_TRIM_SECONDS * STEPS_PER_SECOND
    reason = None

    for t in range(n_steps):
        bng.control.step(1)
        veh.sensors.poll()
        st = veh.state
        el = veh.sensors["electrics"]

        vx, vy, heading = body_frame(st)
        sim_time = st.get("time")
        dt = C.DT
        if prev_time is not None and sim_time is not None and sim_time > prev_time:
            dt = sim_time - prev_time
        r = 0.0 if prev_heading is None else wrap_angle(heading - prev_heading) / dt

        # Acceleration horizontale (repere monde), pour reperer les chocs.
        vel = st["vel"]
        acc_h = 0.0
        if prev_vel is not None:
            acc_h = math.hypot(vel[0] - prev_vel[0], vel[1] - prev_vel[1]) / dt
        prev_heading, prev_time, prev_vel = heading, sim_time, vel

        throttle = float(el.get("throttle", 0.0))
        brake = float(el.get("brake", 0.0))
        pos = st["pos"]
        rows.append([run_id, t, vx, vy, r, float(el.get("steering", 0.0)), throttle - brake,
                     throttle, brake,
                     float(el.get("accXSmooth", 0.0)), float(el.get("accYSmooth", 0.0)),
                     float(el.get("wheelspeed", 0.0)), aggression, speed_limit, spawn_id,
                     pos[0], pos[1], pos[2]])

        if t > 1 and acc_h > CRASH_ACC:
            rows = rows[:max(0, len(rows) - trim_steps)]
            reason = f"choc ({acc_h:.0f} m/s2) a t={t / STEPS_PER_SECOND:.1f}s"
            break
        stuck_steps = stuck_steps + 1 if (t > warmup and abs(vx) < STUCK_SPEED) else 0
        if stuck_steps >= stuck_limit:
            rows = rows[:max(0, len(rows) - stuck_limit - trim_steps)]
            reason = f"voiture bloquee a t={t / STEPS_PER_SECOND:.1f}s"
            break

    return rows, aggression, speed_limit, reason


def collect(n_runs, run_seconds):
    if not SPAWNS:
        sys.exit("SPAWNS est vide : ajoute au moins une position (x, y, z) en haut du fichier.")
    from beamngpy import BeamNGpy, Scenario, Vehicle
    from beamngpy.sensors import Electrics

    rng = random.Random(C.SEED + 1)
    spawns = [(x, y, z + 0.5) for x, y, z in SPAWNS]   # +0.5 m pour ne pas apparaitre dans le sol

    bng = BeamNGpy("localhost", 25252, home=HOME)
    bng.open()
    scenario = Scenario(LEVEL, "rally_jepa_dirt")
    veh = Vehicle("ego", model=VEHICLE_MODEL, license="JEPA")
    veh.sensors.attach("electrics", Electrics())
    scenario.add_vehicle(veh, pos=spawns[0], rot_quat=(0, 0, 0, 1))
    scenario.make(bng)
    bng.settings.set_deterministic(STEPS_PER_SECOND)
    bng.scenario.load(scenario)
    bng.scenario.start()

    can_teleport = True
    total = 0
    try:
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            for run_id in range(n_runs):
                spawn_id = rng.randrange(len(spawns))
                if can_teleport:
                    try:
                        veh.teleport(spawns[spawn_id], reset=True)
                    except Exception as e:
                        print("  (teleport impossible, tous les runs partiront du 1er point :", e, ")")
                        can_teleport = False
                if not can_teleport:
                    spawn_id = 0
                    bng.scenario.restart()
                t0 = time.time()
                rows, agg, lim, reason = record_run(bng, veh, run_id, spawn_id,
                                                    run_seconds * STEPS_PER_SECOND, rng)
                w.writerows(rows)
                f.flush()
                total += len(rows)
                msg = f"  run {run_id} : depart {spawn_id}, {len(rows)} pas gardes, agressivite {agg}, " \
                      f"vitesse max {lim} m/s, {time.time() - t0:.0f} s reelles"
                print(msg + (f"  [coupe : {reason}]" if reason else ""), flush=True)
    finally:
        bng.close()
    print(f"[collect] {total} lignes -> {OUT_CSV}")


if __name__ == "__main__":
    if "--test" in sys.argv:
        collect(n_runs=2, run_seconds=20)
    else:
        collect(n_runs=N_RUNS, run_seconds=RUN_SECONDS)
