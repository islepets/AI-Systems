from owlready2 import *
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
import random


class DroneOntology:

    def __init__(self, filepath="lab3.rdf"):
        self.filepath = filepath
        self.onto = get_ontology(filepath).load()

        self.drone = self.onto.MyDrone
        self.weather = self.onto.CurrentWeather
        self.obstacle = self.onto.NearObstacle

    def get_state(self):
        return {
            'speed': self.drone.hasSpeed[0] if self.drone.hasSpeed else 4.9,
            'altitude': self.drone.hasAltitude[0] if self.drone.hasAltitude else 19.9,
            'wind': self.weather.hasWind[0] if self.weather.hasWind else 0.9,
            'visibility': self.weather.hasVisibity[0] if self.weather.hasVisibity else 99.9,
            'distance': self.obstacle.hasDistance[0] if self.obstacle.hasDistance else 999.9
        }

    def update(self, speed, altitude, wind, visibility, distance):
        self.drone.hasSpeed = [speed]
        self.drone.hasAltitude = [altitude]
        self.weather.hasWind = [wind]
        self.weather.hasVisibity = [visibility]
        self.obstacle.hasDistance = [distance]
        self.onto.save(self.filepath)

    def get_rule(self):
        rules = []
        for rule in self.drone.hasRule:
            priority = rule.hasPriority[0] if rule.hasPriority else 1
            rules.append({
                'name': rule.name,
                'priority': priority
            })
        return sorted(rules, key=lambda x: x['priority'])

    def get_rules_priorities(self):
        rules = self.get_rule()
        return {rule['name']: rule['priority'] for rule in rules}


class Fuzzy:

    def __init__(self):

        self.universe_speed = np.arange(0, 16, 0.5)
        self.universe_altitude = np.arange(0, 61, 0.5)
        self.universe_wind = np.arange(0, 31, 0.5)
        self.universe_visibility = np.arange(0, 101, 1)
        self.universe_distance = np.arange(0, 1001, 1)
        self.universe_output = np.arange(0, 20, 0.5)

        self.speed = ctrl.Antecedent(self.universe_speed, 'speed')
        self.altitude = ctrl.Antecedent(self.universe_altitude, 'altitude')
        self.wind = ctrl.Antecedent(self.universe_wind, 'wind')
        self.visibility = ctrl.Antecedent(self.universe_visibility, 'visibility')
        self.distance = ctrl.Antecedent(self.universe_distance, 'distance')

        self.recommended_speed = ctrl.Consequent(self.universe_output, 'recommended_speed')

        self.speed['low'] = fuzz.trimf(self.universe_speed, [0, 0, 5])
        self.speed['normal'] = fuzz.trimf(self.universe_speed, [5, 8, 11])
        self.speed['fast'] = fuzz.trimf(self.universe_speed, [10, 13, 15])

        self.altitude['low'] = fuzz.trimf(self.universe_altitude, [0, 10, 15])
        self.altitude['medium'] = fuzz.trimf(self.universe_altitude, [13, 28, 31])
        self.altitude['high'] = fuzz.trimf(self.universe_altitude, [30, 43, 59])

        self.wind['calm'] = fuzz.trimf(self.universe_wind, [0, 0, 5])
        self.wind['moderate'] = fuzz.trimf(self.universe_wind, [5, 8, 11])
        self.wind['strong'] = fuzz.trimf(self.universe_wind, [10, 13, 15])

        self.visibility['poor'] = fuzz.trimf(self.universe_visibility, [0, 0, 30])
        self.visibility['moderate'] = fuzz.trimf(self.universe_visibility, [20, 50, 80])
        self.visibility['good'] = fuzz.trimf(self.universe_visibility, [70, 90, 100])

        self.distance['critical'] = fuzz.trimf(self.universe_distance, [0, 0, 50])
        self.distance['close'] = fuzz.trimf(self.universe_distance, [40, 120, 230])
        self.distance['far'] = fuzz.trimf(self.universe_distance, [220, 340, 560])

        self.recommended_speed['very_slow'] = fuzz.trimf(self.universe_output, [0, 0, 3])
        self.recommended_speed['slow'] = fuzz.trimf(self.universe_output, [2, 5, 8])
        self.recommended_speed['moderate'] = fuzz.trimf(self.universe_output, [6, 9, 12])
        self.recommended_speed['fast'] = fuzz.trimf(self.universe_output, [10, 13, 16])
        self.recommended_speed['very_fast'] = fuzz.trimf(self.universe_output, [14, 17, 20])

        rule1 = ctrl.Rule(self.speed['fast'] & self.distance['close'], self.recommended_speed['slow'])
        rule2 = ctrl.Rule(self.distance['critical'], self.recommended_speed['very_slow'])
        rule3 = ctrl.Rule(self.wind['strong'], self.recommended_speed['slow'])
        rule4 = ctrl.Rule(self.wind['moderate'] & self.visibility['poor'], self.recommended_speed['slow'])
        rule5 = ctrl.Rule(self.visibility['poor'], self.recommended_speed['slow'])
        rule6 = ctrl.Rule(self.visibility['good'] & self.wind['calm'], self.recommended_speed['fast'])
        rule7 = ctrl.Rule(self.altitude['low'], self.recommended_speed['slow'])
        rule8 = ctrl.Rule(self.visibility['good'] & self.wind['calm'] & self.distance['far'], self.recommended_speed['very_fast'])

        self.control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8])
        self.controller = ctrl.ControlSystemSimulation(self.control_system)

    def compute(self, current_speed, altitude, wind, visibility, distance):
        self.controller.input['speed'] = current_speed
        self.controller.input['altitude'] = altitude
        self.controller.input['wind'] = wind
        self.controller.input['visibility'] = visibility
        self.controller.input['distance'] = distance

        self.controller.compute()

        if self.controller.output.get('recommended_speed'):
            return float(self.controller.output['recommended_speed'])
        else:
            return current_speed


class DroneSimulator:
    def __init__(self, ontology_path="lab3.rdf"):
        self.ontology = DroneOntology(ontology_path)
        self.fuzzy = Fuzzy()
        self.time = 0
        self.history = {
            'time': [],
            'speed': [],
            'altitude': [],
            'wind': [],
            'visibility': [],
            'distance': [],
            'recommended_speed': []
        }

    def generate_dynamic_environment(self, scenario_type="normal"):
        state = self.ontology.get_state()

        if scenario_type == "normal":
            state['wind'] = max(0, min(20, state['wind'] + random.uniform(-1, 1)))
            state['visibility'] = max(0, min(100, state['visibility'] + random.uniform(-5, 5)))
            state['distance'] = max(0, min(1000, state['distance'] + random.uniform(-10, 10)))

        elif scenario_type == "windy":
            state['wind'] = min(20, state['wind'] + random.uniform(0, 2))
            state['visibility'] = max(0, min(100, state['visibility'] + random.uniform(-3, 1)))

        elif scenario_type == "foggy":
            state['wind'] = max(0, min(20, state['wind'] + random.uniform(-0.5, 1)))
            state['visibility'] = max(0, state['visibility'] - random.uniform(0, 3))

        elif scenario_type == "obstacle_approaching":
            state['distance'] = max(0, state['distance'] - random.uniform(5, 15))
            state['wind'] = max(0, min(20, state['wind'] + random.uniform(-0.5, 1)))

        return state

    def update_drone_state(self, state):
        self.ontology.update(
            speed=state['speed'],
            altitude=state['altitude'],
            wind=state['wind'],
            visibility=state['visibility'],
            distance=state['distance']
        )

    def simulate_step(self, scenario_type="normal"):
        state = self.ontology.get_state()

        new_state = self.generate_dynamic_environment(scenario_type)

        recommended_speed = self.fuzzy.compute(
            state['speed'],
            new_state['altitude'],
            new_state['wind'],
            new_state['visibility'],
            new_state['distance']
        )

        speed_diff = recommended_speed - state['speed']
        new_speed = state['speed'] + speed_diff * 0.3

        new_speed = max(0, min(20, new_speed))

        new_state['speed'] = new_speed

        self.history['time'].append(self.time)
        self.history['speed'].append(new_speed)
        self.history['altitude'].append(new_state['altitude'])
        self.history['wind'].append(new_state['wind'])
        self.history['visibility'].append(new_state['visibility'])
        self.history['distance'].append(new_state['distance'])
        self.history['recommended_speed'].append(recommended_speed)

        self.update_drone_state(new_state)

        self.time += 1

        return new_state, recommended_speed

    def run_scenario(self, scenario_type, steps=100, verbose=True):

        print(scenario_type.upper())

        self.history = {key: [] for key in self.history.keys()}
        self.time = 0

        initial_state = self.ontology.get_state()
        initial_state['wind'] = random.uniform(0, 5)
        initial_state['visibility'] = random.uniform(70, 100)
        initial_state['distance'] = random.uniform(300, 500)
        self.update_drone_state(initial_state)

        speeds = []
        recommendations = []

        for step in range(steps):
            state, recommended = self.simulate_step(scenario_type)
            speeds.append(state['speed'])
            recommendations.append(recommended)

        avg_speed = np.mean(speeds)
        min_speed = min(speeds)
        max_speed = max(speeds)

        print(f"Средняя скорость: {avg_speed:.2f} м/с")
        print(f"Мин. скорость: {min_speed:.2f} м/с")
        print(f"Макс. скорость: {max_speed:.2f} м/с")

        min_distance = min(self.history['distance']) if self.history['distance'] else 1000
        return self.history

    def plot_results(self, scenario_type):
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle(f'Результаты симуляции: {scenario_type}', fontsize=14, fontweight='bold')

        axes[0, 0].plot(self.history['time'], self.history['speed'], 'b-', linewidth=2, label='Фактическая')
        axes[0, 0].plot(self.history['time'], self.history['recommended_speed'], 'r--', linewidth=2,
                        label='Рекомендуемая')
        axes[0, 0].set_xlabel('Время (шаги)')
        axes[0, 0].set_ylabel('Скорость (м/с)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 20)

        axes[0, 1].plot(self.history['time'], self.history['altitude'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Время (шаги)')
        axes[0, 1].set_ylabel('Высота (м)')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(self.history['time'], self.history['wind'], 'orange', linewidth=2)
        axes[1, 0].set_xlabel('Время (шаги)')
        axes[1, 0].set_ylabel('Скорость ветра (м/с)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 25)

        axes[1, 1].plot(self.history['time'], self.history['visibility'], 'purple', linewidth=2)
        axes[1, 1].set_xlabel('Время (шаги)')
        axes[1, 1].set_ylabel('Видимость (%)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 100)

        axes[2, 0].plot(self.history['time'], self.history['distance'], 'brown', linewidth=2)
        axes[2, 0].set_xlabel('Время (шаги)')
        axes[2, 0].set_ylabel('Расстояние до препятствия (м)')
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Критическое расстояние')
        axes[2, 0].legend()

        diff = np.abs(np.array(self.history['speed']) - np.array(self.history['recommended_speed']))
        axes[2, 1].plot(self.history['time'], diff, 'r-', linewidth=2)
        axes[2, 1].set_xlabel('Время (шаги)')
        axes[2, 1].set_ylabel('Отклонение скорости (м/с)')
        axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def main():

    simulator = DroneSimulator("lab3.rdf")
    rules_priorities = simulator.ontology.get_rules_priorities()

    scenarios = [
        ("normal", "Нормальные погодные условия"),
        ("windy", "Усиление ветра"),
        ("foggy", "Ухудшение видимости"),
        ("obstacle_approaching", "Приближение препятствия")
    ]

    results = {}
    for scenario_type, description in scenarios:
        history = simulator.run_scenario(scenario_type, steps=100, verbose=True)
        results[scenario_type] = history
        simulator.plot_results(scenario_type)

if __name__ == '__main__':
    main()