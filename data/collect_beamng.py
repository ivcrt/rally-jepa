# data/collect_beamng.py
# Collecte de vraies donnees BeamNG.tech 0.39 + BeamNGpy 1.36.
# Ecrit le meme format de raw CSV que le synthetique, puis :
#     python data/make_dataset.py --no-synth
#
# Fonctionnement :
#   - la carte est chargee UNE seule fois (c'est le plus long) ;
#   - chaque "run" = l'IA de BeamNG conduit pendant RUN_SECONDS avec une
#     agressivite tiree au hasard, puis on remet la voiture au depart ;
#   - si la voiture reste bloquee (crash), on coupe le run et on jette les
#     dernieres secondes, qui contiennent le choc (dynamique de collision,
#     pas dynamique du vehicule).
#
# Test rapide (2 runs de 20 s) :  python data/collect_beamng.py --test

import os
import sys
import csv
import math
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

# --- A MODIFIER : dossier ou tu as dezippe BeamNG.tech (celui qui contient Bin64) ---
HOME = r"D:\BeamNG.tech"

LEVEL = "west_coast_usa"
SPAWN_POS = (-717, 101, 118)
SPAWN_ROT = (0, 0, 0.3826834, 0.9238795)
VEHICLE_MODEL = "etk800"

STEPS_PER_SECOND = round(1 / C.DT)   # 50 pas/s = DT de 0.02 s
N_RUNS = 30
RUN_SECONDS = 60
WARMUP_SECONDS = 5        # temps laisse a l'IA pour demarrer
STUCK_SPEED = 1.0         # m/s : en dessous, la voiture est consideree a l'arret
STUCK_SECONDS = 2         # arretee aussi longtemps apres le demarrage = run coupe
CRASH_TRIM_SECONDS = 2    # secondes jetees avant l'arret (le choc)

# Colonnes ecrites. Les 5 premieres apres run_id,t sont celles de config.py ;
# les autres sont gardees pour plus tard (verifs, features en plus).
EXTRA_COLS = ["throttle", "brake", "acc_x", "acc_y", "wheelspeed", "aggression", "speed_limit"]
HEADER = ["run_id", "t"] + C.STATE_COLS + C.ACTION_COLS + EXTRA_COLS


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def wrap_angle(a):
    """Ramene un angle dans [-pi, pi] (le cap saute de +pi a -pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def body_frame(state):
    """Vitesses dans le repere de la voiture (x vers l'avant, y vers la gauche)."""
    vel = state["vel"]
    fwd = state["dir"]
    up = state["up"]
    left = cross(up, fwd)          # z vers le haut, x vers l'avant -> y vers la gauche
    vx = dot(vel, fwd)
    vy = dot(vel, left)
    heading = math.atan2(fwd[1], fwd[0])   # cap dans le plan horizontal
    return vx, vy, heading


def record_run(bng, veh, run_id, n_steps, rng):
    """Fait conduire l'IA et renvoie la liste des lignes de ce run."""
    aggression = round(rng.uniform(0.3, 1.0), 2)
    speed_limit = round(rng.uniform(12, 30), 1)   # m/s, soit environ 45 a 110 km/h
    # "random" : l'IA choisit au hasard aux intersections, donc chaque run
    # prend un trajet different (en "span" elle refaisait toujours la meme autoroute).
    veh.ai.set_mode("random")
    veh.ai.set_aggression(aggression)
    try:
        veh.ai.set_speed(speed_limit, mode="limit")   # plafond, pas vitesse imposee
    except Exception as e:
        print("  (limite de vitesse non appliquee :", e, ")")

    rows = []
    prev_heading, prev_time = None, None
    stuck_steps = 0
    warmup = WARMUP_SECONDS * STEPS_PER_SECOND
    stuck_limit = STUCK_SECONDS * STEPS_PER_SECOND

    for t in range(n_steps):
        bng.control.step(1)
        veh.sensors.poll()
        st = veh.state
        el = veh.sensors["electrics"]

        vx, vy, heading = body_frame(st)
        sim_time = st.get("time")

        # Vitesse de lacet = derivee du cap.
        if prev_heading is None:
            r = 0.0
        else:
            dt = (sim_time - prev_time) if (sim_time is not None and prev_time is not None) else C.DT
            dt = dt if dt > 0 else C.DT
            r = wrap_angle(heading - prev_heading) / dt
        prev_heading, prev_time = heading, sim_time

        throttle = float(el.get("throttle", 0.0))
        brake = float(el.get("brake", 0.0))
        steering = float(el.get("steering", 0.0))   # angle du volant en degres (+ = gauche), pas l'angle des roues

        rows.append([run_id, t, vx, vy, r, steering, throttle - brake,
                     throttle, brake,
                     float(el.get("accXSmooth", 0.0)), float(el.get("accYSmooth", 0.0)),
                     float(el.get("wheelspeed", 0.0)), aggression, speed_limit])

        # Detection de crash : trop longtemps a l'arret apres le demarrage.
        if t > warmup and abs(vx) < STUCK_SPEED:
            stuck_steps += 1
        else:
            stuck_steps = 0
        if stuck_steps >= stuck_limit:
            trim = stuck_limit + CRASH_TRIM_SECONDS * STEPS_PER_SECOND
            rows = rows[:max(0, len(rows) - trim)]
            print(f"  run {run_id} : voiture bloquee a t={t / STEPS_PER_SECOND:.1f}s, run coupe")
            break

    print(f"  run {run_id} : {len(rows)} pas gardes, agressivite {aggression}, vitesse max {speed_limit} m/s")
    return rows


def collect(n_runs, run_seconds):
    from beamngpy import BeamNGpy, Scenario, Vehicle
    from beamngpy.sensors import Electrics

    rng = random.Random(C.SEED)
    bng = BeamNGpy("localhost", 25252, home=HOME)
    bng.open()

    scenario = Scenario(LEVEL, "rally_jepa_collect")
    veh = Vehicle("ego", model=VEHICLE_MODEL, license="JEPA")
    veh.sensors.attach("electrics", Electrics())
    scenario.add_vehicle(veh, pos=SPAWN_POS, rot_quat=SPAWN_ROT)
    scenario.make(bng)

    bng.settings.set_deterministic(STEPS_PER_SECOND)
    bng.scenario.load(scenario)          # long la premiere fois
    bng.scenario.start()

    os.makedirs(C.DATA_DIR, exist_ok=True)
    total = 0
    try:
        with open(C.RAW_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            for run_id in range(n_runs):
                if run_id > 0:
                    bng.scenario.restart()   # remet la voiture au depart, sans recharger la carte
                rows = record_run(bng, veh, run_id, run_seconds * STEPS_PER_SECOND, rng)
                w.writerows(rows)
                f.flush()                    # rien de perdu si le script plante en route
                total += len(rows)
    finally:
        bng.close()

    print(f"[collect] {total} lignes -> {C.RAW_CSV}")
    print("Ensuite : python data/make_dataset.py --no-synth")


if __name__ == "__main__":
    if "--test" in sys.argv:
        collect(n_runs=2, run_seconds=20)
    else:
        collect(n_runs=N_RUNS, run_seconds=RUN_SECONDS)
