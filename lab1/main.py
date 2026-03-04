'''

Провести эксперименты по разным способам скрещивания (не менее 3-х), разным способам мутирования (не менее трех). Результат отобразить в виде графиков
Моделирование данных производить на основе максимально правдоподобных данных.

На языке Python разработайте скрипт, который с помощью генетического алгоритма и полного перебора решает следующую задачу.
Дано N наименований продуктов, для каждого из которых известно m характеристик.
Необходимо получить самый дешевый рацион из k наименований, удовлетворяющий заданным медицинским нормам для каждой из m характеристик.

'''

import random
import time
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms


products = [
    {"name": "Хлеб ржаной", "price": 25, "characteristics": [4.7, 0.7, 49.8]},
    {"name": "Хлеб 'Бородинский'", "price": 30, "characteristics": [6.9, 1.3, 40.9]},
    {"name": "Хлеб зерновой", "price": 35, "characteristics": [8.6, 1.4, 45.1]},
    {"name": "Хлеб цельнозерновой из смеси злаков", "price": 40, "characteristics": [13.3, 4.2, 43.3]},
    {"name": "Хлебцы Dr.Körner 'Семь злаков'", "price": 80, "characteristics": [10, 2.5, 72]},
    {"name": "Хлебцы Dr.Körner 'Бородинские'", "price": 85, "characteristics": [11, 3.5, 22]},
    {"name": "Сдоба", "price": 40, "characteristics": [7.4, 2.2, 52.9]},
    {"name": "Сухари из пшеничной муки", "price": 50, "characteristics": [11.2, 1.4, 72.4]},
    {"name": "Мука пшеничная из твердых сортов в/с", "price": 45, "characteristics": [10.8, 1.3, 69.9]},
    {"name": "Мука пшеничная в/с", "price": 40, "characteristics": [10.3, 1.1, 70.6]},
    {"name": "Мука пшеничная 1 сорта", "price": 35, "characteristics": [10.6, 1.3, 69]},
    {"name": "Мука пшеничная 2 сорта", "price": 30, "characteristics": [11.6, 1.8, 64.8]},
    {"name": "Мука ржаная сеяная", "price": 35, "characteristics": [6.9, 1.4, 66.3]},
    {"name": "Мука ржаная обдирная", "price": 30, "characteristics": [8.9, 1.7, 61.8]},
    {"name": "Мука ржаная обойная", "price": 25, "characteristics": [10.7, 1.9, 58.5]},
    {"name": "Макаронные изделия из муки твердых сортов в/с", "price": 60, "characteristics": [11, 1.3, 70.5]},
    {"name": "Макаронные изделия из муки твердых сортов в/с, вареные", "price": 20, "characteristics": [3.6, 0.4, 20]},
    {"name": "Крупа гречневая ядрица", "price": 50, "characteristics": [12.6, 3.3, 57.1]},
    {"name": "Гречка вареная", "price": 15, "characteristics": [4.1, 1, 18.5]},
    {"name": "Рис белый шлифованный", "price": 45, "characteristics": [7, 1, 74]},
    {"name": "Рис белый шлифованный вареный", "price": 15, "characteristics": [2.3, 0.2, 28.7]},
    {"name": "Пшено", "price": 30, "characteristics": [11.5, 3.3, 66.5]},
    {"name": "Пшено вареное", "price": 10, "characteristics": [3.5, 1.1, 23.6]},
    {"name": "Крупа овсяная", "price": 35, "characteristics": [12.3, 6.1, 59.5]},
    {"name": "Овсяные хлопья", "price": 40, "characteristics": [12.3, 6.2, 61.8]},
    {"name": "Каша из овсяных хлопьев на воде", "price": 10, "characteristics": [3, 1.7, 15]},
    {"name": "Перловая крупа", "price": 25, "characteristics": [9.3, 1.1, 66.9]},
    {"name": "Ячневая крупа", "price": 25, "characteristics": [10, 1.3, 65.4]},
    {"name": "Горох целый шлифованный", "price": 40, "characteristics": [22, 2, 57]},
    {"name": "Чечевица", "price": 55, "characteristics": [24.6, 1.1, 63.4]},
    {"name": "Нут", "price": 60, "characteristics": [20.5, 4.3, 63]},
    {"name": "Соя зерно", "price": 45, "characteristics": [36.7, 17.3, 17.3]},
    {"name": "Молоко 1,5% жирности", "price": 40, "characteristics": [3, 1.5, 4.8]},
    {"name": "Молоко 2,5%", "price": 42, "characteristics": [2.9, 2.5, 4.8]},
    {"name": "Молоко 3,2%", "price": 45, "characteristics": [2.9, 3.2, 4.7]},
    {"name": "Сливки 10%", "price": 70, "characteristics": [2.7, 10, 4.4]},
    {"name": "Творог нежирный", "price": 80, "characteristics": [22, 0.6, 3.3]},
    {"name": "Творог 5%", "price": 85, "characteristics": [21, 5, 3]},
    {"name": "Сметана 20%", "price": 90, "characteristics": [2.5, 20, 3.4]},
    {"name": "Кефир 1%", "price": 45, "characteristics": [3, 1, 4]},
    {"name": "Сыр 'Адыгейский'", "price": 200, "characteristics": [19.8, 19.8, 1.5]},
    {"name": "Сыр 'Пармезан'", "price": 400, "characteristics": [37.8, 27.3, 3.4]},
    {"name": "Йогурт 'Активиа'", "price": 60, "characteristics": [3.8, 2.9, 14]},
    {"name": "Говядина 1 категории", "price": 300, "characteristics": [18.6, 16, 0]},
    {"name": "Говядина, вырезка", "price": 400, "characteristics": [22.2, 7.1, 0]},
    {"name": "Свинина мясная", "price": 250, "characteristics": [14.3, 33.3, 0]},
    {"name": "Куриная грудка", "price": 150, "characteristics": [23.6, 1.9, 0]},
    {"name": "Индейка, грудка", "price": 200, "characteristics": [23.6, 1.5, 0]},
    {"name": "Печень говяжья", "price": 120, "characteristics": [17.9, 3.7, 5.3]},
    {"name": "Кролик", "price": 350, "characteristics": [21.2, 11, 0]},
    {"name": "Колбаса 'Докторская'", "price": 250, "characteristics": [12.8, 22.2, 1.5]},
    {"name": "Сосиски молочные", "price": 200, "characteristics": [11, 23.9, 0.4]},
    {"name": "Горбуша", "price": 180, "characteristics": [20.5, 6.5, 0]},
    {"name": "Кальмар", "price": 250, "characteristics": [18, 2.2, 2]},
    {"name": "Креветки", "price": 300, "characteristics": [17, 2.2, 0.6]},
    {"name": "Минтай", "price": 120, "characteristics": [15.9, 0.9, 0]},
    {"name": "Семга", "price": 400, "characteristics": [21, 6, 0]},
    {"name": "Икра красная", "price": 800, "characteristics": [32, 15, 0]},
    {"name": "Яйцо куриное", "price": 60, "characteristics": [12.7, 10.9, 0.7]},
    {"name": "Масло подсолнечное", "price": 80, "characteristics": [0, 99.9, 0]},
    {"name": "Масло сливочное", "price": 150, "characteristics": [0.5, 82.5, 0.8]},
    {"name": "Майонез", "price": 100, "characteristics": [1.4, 67, 2.6]},
    {"name": "Картофель", "price": 20, "characteristics": [2, 0.4, 16.3]},
    {"name": "Морковь", "price": 25, "characteristics": [1.3, 0.1, 6.9]},
    {"name": "Капуста белокочанная", "price": 15, "characteristics": [1.8, 0.1, 4.7]},
    {"name": "Помидоры грунтовые", "price": 80, "characteristics": [1.1, 0.2, 3.8]},
    {"name": "Огурцы грунтовые", "price": 60, "characteristics": [0.8, 0.1, 2.5]},
    {"name": "Лук репчатый", "price": 20, "characteristics": [1.4, 0.2, 8.2]},
    {"name": "Чеснок", "price": 100, "characteristics": [6.5, 0.5, 29.9]},
    {"name": "Банан", "price": 70, "characteristics": [1.5, 0.5, 21]}
]

norma_bmr = [80, 55, 200]

# -----------------------------------------------------------------------------------------------------------------------
# Полный перебор
# -----------------------------------------------------------------------------------------------------------------------
'''
def full(products, norma_bmr, k=5):
    n = len(products)
    best_combo = None
    best_score = float('inf')
    best_price = None

    for combo in combinations(range(n), k):
        total_nut = np.sum([products[i]['characteristics'] for i in combo], axis=0)
        total_price = np.sum([products[i]['price'] for i in combo])

        dev = np.sum(np.abs(total_nut - np.array(norma_bmr)))

        score = dev + 0.01 * total_price

        if score < best_score:
            best_score = score
            best_combo = [products[i]['name'] for i in combo]
            best_price = total_price

    return best_combo, best_price


start = time.time()
best_combo, best_price = full(products, norma_bmr)
print('Список лучших продуктов: ' + ', '.join(best_combo))
print(f'Лучшая стоимость: {best_price}')
print(f'Время выполнения алгоритма: {(time.time()-start):.2f} с')
'''
# -----------------------------------------------------------------------------------------------------------------------
# Генетический алгоритм
# -----------------------------------------------------------------------------------------------------------------------

N = len(products)
K = 5

POPULATION_SIZE = 200       # количество особей в популяции
GEN_COUNT = 50              # количество поколений
P_CROSS = 0.72              # вероятность скрещивания
P_MUT = 0.31                # вероятность мутации

random.seed(42)
np.random.seed(42)

creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox() # контейнер для операторов алгоритма

def fitness_fun(individual):
    nut = np.zeros(3) # массив нутриентов
    total_price = 0   # текущая стоимость
    for i, gene in enumerate(individual): # проходимся по всей хромосоме
        if gene == 1:  # если ген равен 1 ( то есть включен)
            nut += np.array(products[i]['characteristics']) # добавляем в нутриенты
            total_price += products[i]['price'] # считаем стоимость

    if np.sum(individual) != K: # если сумма генов неравно к, тогда вешаем штраф
        return 10000,

    if any(nut < np.array(norma_bmr)):  # если не хватает нутриенотов, тогда вешаем штраф
        return 5000 + np.sum(np.maximum(0, np.array(norma_bmr) - nut)) * 100,

    dev = np.sum(np.abs(nut - np.array(norma_bmr))) # рассчитываем отклонение от нормы (по модулю)
    score = dev + 0.01 * total_price
    return score,

def create_individual(): # функция создания особи
    ind = [0] * N # создаем массив из из 70 генов
    index = random.sample(range(N), K) # рандомно заполняем позиции в количестве к
    for i in index:
        ind[i] = 1
    return creator.Individual(ind)

toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitness_fun)
toolbox.register("select", tools.selTournament, tournsize=5)

# Определяем методы скрещивания
crossover_methods = [
    ("Одноточечное", tools.cxOnePoint),
    ("Двуточечное", tools.cxTwoPoint),
    ("Равномерное", lambda ind1, ind2: tools.cxUniform(ind1, ind2, indpb=0.5)),
]

def mutation_random_reset(individual, indpb=0.1):
    if random.random() < indpb: # вероятность мутации
        num_to_change = random.randint(1, min(3, K)) # сколько пар меняем
        current = [i for i, val in enumerate(individual) if val == 1] # индексы выбранных продуктов
        not_current = [i for i, val in enumerate(individual) if val == 0] # индексы невыбранных продуктов

        if len(current) >= num_to_change and len(not_current) >= num_to_change: # достатончо ли продуктов для обмена
            to_remove = random.sample(current, num_to_change) # что убираем
            to_add = random.sample(not_current, num_to_change) # что добавляем
            # обмен
            for i in to_remove:
                individual[i] = 0
            for i in to_add:
                individual[i] = 1

    return individual,


mutation_methods = [
    ("Инверсия", lambda ind: tools.mutFlipBit(ind, indpb=0.1)),
    ("Перестановка", lambda ind: tools.mutShuffleIndexes(ind, indpb=0.1)),
    ("Случайная", lambda ind: mutation_random_reset(ind, indpb=0.1))
]

def run_genetic_algorithm(cx_method, mut_method, verbose=False):
    exp_toolbox = base.Toolbox()

    #регистрация всех операторов
    exp_toolbox.register("individual", create_individual)
    exp_toolbox.register("population", tools.initRepeat, list, exp_toolbox.individual)
    exp_toolbox.register("evaluate", fitness_fun)
    exp_toolbox.register("select", tools.selTournament, tournsize=5)
    exp_toolbox.register("mate", cx_method)
    exp_toolbox.register("mutate", mut_method)

    population = exp_toolbox.population(n=POPULATION_SIZE) # создание начальной популяции

    stats = tools.Statistics(lambda ind: ind.fitness.values) # настройка статистики
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    population, logbook = algorithms.eaSimple( # запуск эволюции
        population, exp_toolbox,
        cxpb=P_CROSS,
        mutpb=P_MUT,
        ngen=GEN_COUNT,
        stats=stats,
        verbose=verbose
    )
    # поиск лучшей особи
    best_individual = tools.selBest(population, 1)[0]
    best_fitness = best_individual.fitness.values[0]

    return best_individual, best_fitness, logbook

results = []
for cx_name, cx_fun in crossover_methods:
    for mut_name, mut_fun in mutation_methods:
        exp_name = f"{cx_name} + {mut_name}"

        best_ind, best_fit, logbook = run_genetic_algorithm(cx_fun, mut_fun)

        products_in_ration = [products[i]['name'] for i, val in enumerate(best_ind) if val == 1]
        total_nut = np.sum([products[i]['characteristics'] for i, val in enumerate(best_ind) if val == 1], axis=0)
        total_price = np.sum([products[i]['price'] for i, val in enumerate(best_ind) if val == 1])

        results.append({
            "name": exp_name,
            "logbook": logbook,
            "best_fitness": best_fit,
            "best_individual": best_ind,
            "products": products_in_ration,
            "nutrition": total_nut,
            "price": total_price,
        })


results.sort(key=lambda x: x['best_fitness'])

for res in results:
    generations = range(len(res['logbook'].select("min")))
    min_fitness = res['logbook'].select("min")
    plt.plot(generations, min_fitness, label=res['name'], linewidth=2, alpha=0.8)

plt.xlabel('Поколение', fontsize=12)
plt.ylabel('Фитнес-функция (стоимость)', fontsize=12)
plt.title('Сходимость различных комбинаций операторов', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=9, bbox_to_anchor=(1.35, 1))
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.show()

names = [res['name'] for res in results]
best_values = [res['best_fitness'] for res in results]
colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

bars = plt.barh(range(len(names)), best_values, color=colors, alpha=0.8)
plt.yticks(range(len(names)), names, fontsize=9)
plt.xlabel('Лучшее значение фитнес-функции', fontsize=12)
plt.title('Сравнение эффективности операторов', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')

# Добавляем значения на график
for i, (bar, val) in enumerate(zip(bars, best_values)):
    plt.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height() / 2,
             f'{val:.2f}', va='center', fontsize=8)

plt.tight_layout()
plt.show()

print(f"Комбинация: {results[0]['name']}")
print(f"Стоимость рациона: {results[0]['price']}")