PATH_TO_DLL = "C:\\prjs\\dll\\Aurora__1_model_solution.dll"  # путь к файлу dll'ки
BASELINE = False  # необходимо установить True, если в среде используется baseline управление,
                  # и False, если нейронная сеть

MAX_STEPS = 5000  # Максимально допустимое количество шагов среды
WORK_DIR = "./"  # Путь к директории, в которой будут создаваться папки с параметрами экспериментов
                 # (.csv файлы с координатами точек интереса, навигацией планового бла,...)
N_RANGE = [80, 100]  # интервал количества точек интереса в генерируемой среде
H = 200  # Высота полета беспилотников
LEFT_BOTTOM = [-6000, 0]  # координата нижней левой точки
RIGHT_TOP = [0, 6000]  # координата правой верхней точки
PLAN_UAV_VISION_WIDTH = 0.08  #
PLAN_UAV_VISION_HIGH = 0.08  #
PLAN_UAV_OVERLAP = 0.04  #
FLIGHT_DELAY = False  #
DELAY_STEPS = [400, 800]  #

TIME_PENALTY = 1e-2  # штраф за каждый шаг проведенный в среде
SURVEY_REWARD = 2  # награда за дообследование точки интереса
MC_REWARD = 10  # награда за выполнение задания (обследование всех точек)

DEBUG = False  # включить режим дебага. нужен был для отладки
AGENT_INDEX = 0  # индекс агента. так будет называться папка, в которую будет записываться информация о среде.
                 # (.csv файлы с координатами точек интереса, навигацией планового бла,...)