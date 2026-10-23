# data/collect_slide.py
# Derniere collecte (Utah, terre) : on PROVOQUE la glisse au lieu de l'esperer.
# Differences avec collect_dirt.py :
#   - vitesse IMPOSEE (mode "set") dans 70 % des runs : l'IA doit prendre les
#     virages trop vite pour l'adherence ; les 30 % restants en "limit" pour la variete ;
#   - agressivite elevee (0.7 a 1.0) ;
#   - point de depart n 3 retire (il faisait tous les faux departs la nuit derniere) ;
#   - runs de 20 s ; les 5 premieres secondes (demarrage, demi-tour) sont gardees
#     dans le CSV mais marquees par la colonne t : on les ignorera a l'analyse ;
#   - capteurs en plus : regime, rapport, charge moteur, acceleration verticale,
#     et activite ABS / ESC / antipatinage (pour savoir quand l'electronique a corrige) ;
#   - ecrit data/raw_runs_slide.csv : les deux CSV precedents ne sont PAS ecrases.
#
# Collecte complete :  python data/collect_slide.py
# Test rapide       :  python data/collect_slide.py --test

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
OUT_CSV = os.path.join(C.DATA_DIR, "raw_runs_slide2.csv")   # nuit 4 : n'ecrase pas raw_runs_slide.csv
MAX_DEAD_IN_A_ROW = 5   # IA inactive malgre un rechargement complet, 5 fois de suite = on arrete

# Points de depart sur la terre (l'ancien n 3, (38.80, -96.32, 146.59), est retire).
SPAWNS = [
    (541.12, -517.64, 133.95),
    (489.36, -699.02, 124.39),
    (-338.98, -509.37, 131.32),
    (197.43, 236.09, 184.47),
]

N_RUNS = 400
RUN_SECONDS = 20
WARMUP_SECONDS = 5
STUCK_SPEED = 1.0
STUCK_SECONDS = 2
CRASH_TRIM_SECONDS = 2
CRASH_ACC = 40.0
SET_SPEED_PROB = 0.7          # part des runs a vitesse imposee
SET_SPEED_RANGE = (12, 24)    # m/s imposes (43 a 86 km/h sur la terre)
LIMIT_SPEED_RANGE = (10, 25)  # m/s plafond, pour les autres runs
AGGRESSION_RANGE = (0.7, 1.0)

EXTRA_ELECTRICS = ["rpm", "gear_index", "engine_load", "accZSmooth",
                   "abs_active", "esc_active", "tcs_active"]
EXTRA_COLS = ["throttle", "brake", "acc_x", "acc_y", "wheelspeed",
              "aggression", "speed_target", "speed_mode", "spawn_id", "x", "y", "z"] + EXTRA_ELECTRICS
HEADER = ["run_id", "t"] + C.STATE_COLS + C.ACTION_COLS + EXTRA_COLS


def num(v):
    """Convertit une valeur de capteur en nombre (bool -> 0/1), NaN si impossible."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def record_run(bng, veh, run_id, spawn_id, n_steps, rng):
    aggression = round(rng.uniform(*AGGRESSION_RANGE), 2)
    if rng.random() < SET_SPEED_PROB:
        mode, target = "set", round(rng.uniform(*SET_SPEED_RANGE), 1)
    else:
        mode, target = "limit", round(rng.uniform(*LIMIT_SPEED_RANGE), 1)
    veh.ai.set_mode("random")
    veh.ai.set_aggression(aggression)
    try:
        veh.ai.set_speed(target, mode=mode)
    except Exception as e:
        print(f"  (vitesse {mode} non appliquee : {e})")

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
        vel = st["vel"]
        acc_h = 0.0 if prev_vel is None else math.hypot(vel[0] - prev_vel[0], vel[1] - prev_vel[1]) / dt
        prev_heading, prev_time, prev_vel = heading, sim_time, vel

        throttle = num(el.get("throttle", 0.0))
        brake = num(el.get("brake", 0.0))
        pos = st["pos"]
        rows.append([run_id, t, vx, vy, r, num(el.get("steering", 0.0)), throttle - brake,
                     throttle, brake, num(el.get("accXSmooth")), num(el.get("accYSmooth")),
                     num(el.get("wheelspeed")), aggression, target, mode, spawn_id,
                     pos[0], pos[1], pos[2]] + [num(el.get(k)) for k in EXTRA_ELECTRICS])

        if t > 1 and acc_h > CRASH_ACC:
            rows = rows[:max(0, len(rows) - trim_steps)]
            reason = f"choc ({acc_h:.0f} m/s2) a t={t / STEPS_PER_SECOND:.1f}s"
            break
        stuck_steps = stuck_steps + 1 if (t > warmup and abs(vx) < STUCK_SPEED) else 0
        if stuck_steps >= stuck_limit:
            rows = rows[:max(0, len(rows) - stuck_limit - trim_steps)]
            reason = f"voiture bloquee a t={t / STEPS_PER_SECOND:.1f}s"
            break

    return rows, aggression, mode, target, reason


def collect(n_runs, run_seconds):
    from beamngpy import BeamNGpy, Scenario, Vehicle
    from beamngpy.sensors import Electrics

    rng = random.Random(C.SEED + 2)
    spawns = [(x, y, z + 0.5) for x, y, z in SPAWNS]

    bng = BeamNGpy("localhost", 25252, home=HOME)
    bng.open()
    scenario = Scenario(LEVEL, "rally_jepa_slide")
    veh = Vehicle("ego", model=VEHICLE_MODEL, license="JEPA")
    veh.sensors.attach("electrics", Electrics())
    scenario.add_vehicle(veh, pos=spawns[0], rot_quat=(0, 0, 0, 1))
    scenario.make(bng)
    bng.settings.set_deterministic(STEPS_PER_SECOND)
    bng.scenario.load(scenario)
    bng.scenario.start()

    can_teleport = True
    total = 0
    dead_in_a_row = 0
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
                try:
                    rows, agg, mode, target, reason = record_run(
                        bng, veh, run_id, spawn_id, run_seconds * STEPS_PER_SECOND, rng)
                except Exception as e:
                    # Un run qui plante ne doit pas arreter toute la nuit.
                    print(f"  run {run_id} : erreur ({e}), on passe au suivant", flush=True)
                    continue

                # IA inactive : voiture bloquee ET accelerateur a zero pendant tout le run.
                # (Nuit 3 : apres un choc, l'IA n'a plus jamais conduit.) On recharge le
                # scenario, ce qui recree une voiture neuve, et on jette ce run vide.
                ai_dead = (reason is not None and reason.startswith("voiture bloquee")
                           and sum(row[7] for row in rows) <= 0.01 * max(1, len(rows)))
                if ai_dead:
                    dead_in_a_row += 1
                    print(f"  run {run_id} : IA inactive ({dead_in_a_row}/{MAX_DEAD_IN_A_ROW}), "
                          f"rechargement complet du scenario", flush=True)
                    if dead_in_a_row >= MAX_DEAD_IN_A_ROW:
                        print("  IA toujours inactive apres plusieurs rechargements : arret de la collecte.", flush=True)
                        break
                    bng.scenario.restart()
                    continue
                dead_in_a_row = 0

                w.writerows(rows)
                f.flush()
                total += len(rows)
                msg = f"  run {run_id} : depart {spawn_id}, {len(rows)} pas gardes, {mode} {target} m/s, " \
                      f"agressivite {agg}, {time.time() - t0:.0f} s reelles"
                print(msg + (f"  [coupe : {reason}]" if reason else ""), flush=True)
    finally:
        bng.close()
    print(f"[collect] {total} lignes -> {OUT_CSV}")


if __name__ == "__main__":
    if "--test" in sys.argv:
        collect(n_runs=3, run_seconds=20)
    else:
        collect(n_runs=N_RUNS, run_seconds=RUN_SECONDS)
