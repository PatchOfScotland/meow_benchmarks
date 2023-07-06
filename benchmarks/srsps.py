
from meow_base.core.runner import MeowRunner
from meow_base.patterns import FileEventPattern
from meow_base.recipes import get_recipe_from_notebook

from benchmarks.shared import run_test, SRSES


def single_rule_single_event_sequential(job_count:int, repeats:int, 
        job_counter:int, requested_jobs:int, runtime_start:float)->MeowRunner:
    patterns = {}
    pattern = FileEventPattern(
        f"pattern_one",
        f"testing/*",
        "recipe_two",
        "INPUT_FILE",
        parameters={
            "MAX_COUNT":job_count
        }
    )
    patterns[pattern.name] = pattern

    recipe = get_recipe_from_notebook("recipe_two", "../notebooks/sequential.ipynb")
    
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
        signature=SRSES,
        execution=True,
        print_logging=False
    )
