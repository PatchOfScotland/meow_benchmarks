
import importlib
import os

from multiprocessing import Pipe
from random import shuffle
from shutil import copy

from meow_base.meow_base.conductors import LocalPythonConductor
from meow_base.meow_base.core.runner import MeowRunner
from meow_base.meow_base.functionality.file_io import make_dir, write_file, lines_to_string
from meow_base.meow_base.functionality.requirements import create_python_requirements
from meow_base.meow_base.patterns.file_event_pattern import WatchdogMonitor, \
    FileEventPattern
from meow_base.meow_base.recipes.jupyter_notebook_recipe import PapermillHandler, \
    JupyterNotebookRecipe
from meow_base.tests.shared import TEST_JOB_QUEUE, TEST_JOB_OUTPUT, TEST_MONITOR_BASE, POROSITY_CHECK_NOTEBOOK, SEGMENT_FOAM_NOTEBOOK, GENERATOR_NOTEBOOK, FOAM_PORE_ANALYSIS_NOTEBOOK, IDMC_UTILS_PYTHON_SCRIPT, TEST_DATA, GENERATE_PYTHON_SCRIPT

pattern_check = FileEventPattern(
    "pattern_check", 
    os.path.join("foam_ct_data", "*"), 
    "recipe_check", 
    "input_filename",
    parameters={
        "output_filedir_accepted": 
            os.path.join("{BASE}", "foam_ct_data_accepted"),
        "output_filedir_discarded": 
            os.path.join("{BASE}", "foam_ct_data_discarded"),
        "porosity_lower_threshold": 0.8,
        "utils_path": os.path.join("{BASE}", "idmc_utils_module.py")
    })

pattern_segment = FileEventPattern(
    "pattern_segment",
    os.path.join("foam_ct_data_accepted", "*"),
    "recipe_segment",
    "input_filename",
    parameters={
        "output_filedir": os.path.join("{BASE}", "foam_ct_data_segmented"),
        "input_filedir": os.path.join("{BASE}", "foam_ct_data"),
        "utils_path": os.path.join("{BASE}", "idmc_utils_module.py")
    })

pattern_analysis = FileEventPattern(
    "pattern_analysis",
    os.path.join("foam_ct_data_segmented", "*"),
    "recipe_analysis",
    "input_filename",
    parameters={
        "output_filedir": os.path.join("{BASE}", "foam_ct_data_pore_analysis"),
        "utils_path": os.path.join("{BASE}", "idmc_utils_module.py")
    })


pattern_regenerate_random = FileEventPattern(
    "pattern_regenerate_random",
    os.path.join("foam_ct_data_discarded", "*"),
    "recipe_generator",
    "discarded",
    parameters={
        "dest_dir": os.path.join("{BASE}", "foam_ct_data"),
        "utils_path": os.path.join("{BASE}", "idmc_utils_module.py"),
        "gen_path": os.path.join("{BASE}", "generator.py"),
        "test_data": os.path.join(TEST_DATA, "foam_ct_data"),
        "vx": 32,
        "vy": 32,
        "vz": 32,
        "res": 3/32,
        "chance_good": 1,
        "chance_small": 0,
        "chance_big": 3
    })

patterns = {
    'pattern_check': pattern_check,
    'pattern_segment': pattern_segment,
    'pattern_analysis': pattern_analysis,
    'pattern_regenerate_random': pattern_regenerate_random
}

recipe_check_key, recipe_check_req = create_python_requirements(
    modules=["numpy", "importlib", "matplotlib"])
recipe_check = JupyterNotebookRecipe(
    'recipe_check',
    POROSITY_CHECK_NOTEBOOK, 
    requirements={recipe_check_key: recipe_check_req}
)

recipe_segment_key, recipe_segment_req = create_python_requirements(
    modules=["numpy", "importlib", "matplotlib", "scipy", "skimage"])
recipe_segment = JupyterNotebookRecipe(
    'recipe_segment',
    SEGMENT_FOAM_NOTEBOOK, 
    requirements={recipe_segment_key: recipe_segment_req}
)

recipe_analysis_key, recipe_analysis_req = create_python_requirements(
    modules=["numpy", "importlib", "matplotlib", "scipy", "skimage"])
recipe_analysis = JupyterNotebookRecipe(
    'recipe_analysis',
    FOAM_PORE_ANALYSIS_NOTEBOOK, 
    requirements={recipe_analysis_key: recipe_analysis_req}
)

recipe_generator_key, recipe_generator_req = create_python_requirements(
    modules=["numpy", "matplotlib", "random"])
recipe_generator = JupyterNotebookRecipe(
    'recipe_generator',
    GENERATOR_NOTEBOOK, 
    requirements={recipe_generator_key: recipe_generator_req}           
)

recipes = {
    'recipe_check': recipe_check,
    'recipe_segment': recipe_segment,
    'recipe_analysis': recipe_analysis,
    'recipe_generator': recipe_generator
}

make_dir(TEST_MONITOR_BASE, ensure_clean=True)

runner = MeowRunner(
    WatchdogMonitor(
        TEST_MONITOR_BASE,
        patterns,
        recipes,
        settletime=1
    ), 
    PapermillHandler(),
    LocalPythonConductor(pause_time=2)
)

# Intercept messages between the conductor and runner for testing
conductor_to_test_conductor, conductor_to_test_test = Pipe(duplex=True)
test_to_runner_runner, test_to_runner_test = Pipe(duplex=True)

runner.conductors[0].to_runner_job = conductor_to_test_conductor

for i in range(len(runner.job_connections)):
    _, obj = runner.job_connections[i]

    if obj == runner.conductors[0]:
        runner.job_connections[i] = (test_to_runner_runner, runner.job_connections[i][1])

good = 0
big = 1
small = 0
vx = 32
vy = 32
vz = 32
res = 3/vz
backup_data_dir = os.path.join(TEST_DATA, "foam_ct_data")
make_dir(backup_data_dir)
foam_data_dir = os.path.join(TEST_MONITOR_BASE, "foam_ct_data")
make_dir(foam_data_dir)

write_file(lines_to_string(IDMC_UTILS_PYTHON_SCRIPT), 
    os.path.join(TEST_MONITOR_BASE, "idmc_utils_module.py"))

gen_path = os.path.join(TEST_MONITOR_BASE, "generator.py")
write_file(lines_to_string(GENERATE_PYTHON_SCRIPT), gen_path)

all_data = [1000] * good + [100] * big + [10000] * small
shuffle(all_data)

u_spec = importlib.util.spec_from_file_location("gen", gen_path)
gen = importlib.util.module_from_spec(u_spec)
u_spec.loader.exec_module(gen)

for i, val in enumerate(all_data):
    filename = f"foam_dataset_{i}_{val}_{vx}_{vy}_{vz}.npy"
    backup_file = os.path.join(backup_data_dir, filename)
    if not os.path.exists(backup_file):
        gen.create_foam_data_file(backup_file, val, vx, vy, vz, res)

    target_file = os.path.join(foam_data_dir, filename)
    copy(backup_file, target_file)

runner.start()

loops = 0
idles = 0
while loops < 1200 and idles < 15:
    # Initial prompt
    if conductor_to_test_test.poll(60):
        msg = conductor_to_test_test.recv()
    else:
        break       
    test_to_runner_test.send(msg)

    # Reply
    if test_to_runner_test.poll(15):
        msg = test_to_runner_test.recv()
        if msg == 1:
            idles += 1
        else:
            idles = 0
    else:
        break      
    conductor_to_test_test.send(msg)

    loops += 1

runner.stop()
