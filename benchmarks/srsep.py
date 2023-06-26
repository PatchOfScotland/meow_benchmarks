
from meow_base.core.runner import MeowRunner
from meow_base.functionality.meow import create_parameter_sweep
from meow_base.patterns import FileEventPattern
from meow_base.recipes import get_recipe_from_notebook

from meow_benchmarks.benchmarks.shared import run_test, SRSEP


def single_rule_single_event_parallel(job_count:int, repeats:int, 
        job_counter:int, requested_jobs:int, runtime_start:float)->MeowRunner:
    patterns = {}
    pattern = FileEventPattern(
        f"pattern_one",
        f"testing/*",
        "recipe_one",
        "input",
        sweep=create_parameter_sweep("var", 1, job_count, 1)
    )
    patterns[pattern.name] = pattern

    recipe = get_recipe_from_notebook("recipe_one", "../notebooks/test.ipynb")
    
    recipes = {
        recipe.name: recipe
    }

    return run_test(
        patterns, 
        recipes, 
        1, 
        job_count,
        repeats, 
        job_counter,
        requested_jobs,
        runtime_start,
        signature=SRSEP
    )