
from meow_base.core.runner import MeowRunner
from meow_base.patterns import FileEventPattern
from meow_base.recipes import get_recipe_from_notebook

from benchmarks.shared import run_test, SRME


def single_rule_multiple_events(job_count:int, repeats:int, job_counter:int,
        requested_jobs:int, runtime_start:float)->MeowRunner:
    patterns = {}
    pattern = FileEventPattern(
        f"pattern_one",
        f"testing/*",
        "recipe_one",
        "input"
    )
    patterns[pattern.name] = pattern

    recipe = get_recipe_from_notebook("recipe_one", "../notebooks/test.ipynb")
    
    recipes = {
        recipe.name: recipe
    }

    return run_test(
        patterns, 
        recipes, 
        job_count, 
        job_count,
        repeats, 
        job_counter,
        requested_jobs,
        runtime_start,
        signature=SRME
    )
