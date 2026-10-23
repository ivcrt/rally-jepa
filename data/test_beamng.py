# data/test_beamng.py
# Test de connexion BeamNG.tech + inspection des capteurs, AVANT toute collecte.
# But : savoir ce que TA version renvoie vraiment, pour ecrire collect_beamng.py
# sur des faits et pas sur des suppositions.
#
# Lancer : python data/test_beamng.py
# Puis copier-coller toute la sortie a Claude.

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics

# A MODIFIER : dossier ou tu as dezippe BeamNG.tech (celui qui contient Bin64).
HOME = r"C:\BeamNG.tech"

bng = BeamNGpy("localhost", 25252, home=HOME)   # 25252 = port par defaut
bng.open()                                       # lance le simulateur
print("SYSTEME :", bng.system.get_info())

# Niveau et point de depart de l'exemple officiel BeamNGpy (sur une route).
scenario = Scenario("west_coast_usa", "test_rally_jepa")
veh = Vehicle("ego", model="etk800", license="JEPA")
veh.sensors.attach("electrics", Electrics())
scenario.add_vehicle(veh, pos=(-717, 101, 118), rot_quat=(0, 0, 0.3826834, 0.9238795))
scenario.make(bng)

# Pas de temps fixe : 50 pas par seconde = 0.02 s, comme DT dans config.py.
bng.settings.set_deterministic(50)
bng.scenario.load(scenario)
bng.scenario.start()

# Commande simple : accelerer et tourner un peu, pour avoir du lacet et de la derive.
veh.control(throttle=0.6, steering=0.3)

for t in range(150):              # 150 pas = 3 s
    bng.control.step(1)
    veh.sensors.poll()
    if t in (0, 149):
        print(f"\n--- pas {t} ---")
        print("CLES state :", sorted(veh.state.keys()))
        print("state['vel'] :", veh.state.get("vel"), "| state['dir'] :", veh.state.get("dir"))
        el = veh.sensors["electrics"]
        print("CLES electrics :", sorted(el.keys()))

bng.close()
print("\nTest termine.")
