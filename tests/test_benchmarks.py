
import os
import unittest

from meow_base.patterns.file_event_pattern import FileEventPattern, \
    WatchdogMonitor
from meow_base.recipes.jupyter_notebook_recipe import JupyterNotebookRecipe, \
    PapermillHandler
from meow_base.core.vars import DEFAULT_JOB_OUTPUT_DIR, \
    DEFAULT_JOB_QUEUE_DIR, META_FILE, JOB_FILE, SWEEP_JUMP, SWEEP_START, \
    SWEEP_STOP
from meow_base.conductors.local_python_conductor import LocalPythonConductor
from meow_base.functionality.file_io import read_notebook, rmtree

from meow_benchmarks.benchmarks.mrme import multiple_rules_multiple_events
from meow_benchmarks.benchmarks.mrse import multiple_rules_single_event
from meow_benchmarks.benchmarks.srme import single_rule_multiple_events
from meow_benchmarks.benchmarks.srsep import single_rule_single_event_parallel
from meow_benchmarks.benchmarks.srsps import single_rule_single_event_sequential
from meow_benchmarks.benchmarks.shared import BASE, RESULTS_DIR

def setup():
    pass

def teardown():
    rmtree(DEFAULT_JOB_QUEUE_DIR)
    rmtree(DEFAULT_JOB_OUTPUT_DIR)
    rmtree(BASE)
    rmtree(RESULTS_DIR)

class MultipleRulesMultipleEventsTests(unittest.TestCase):
    def setUp(self)->None:
        super().setUp()
        setup()

    def tearDown(self)->None:
        super().tearDown()
        teardown()

    # Test benchmark runner setup correctly
    def testSetup(self)->None:
        runner = multiple_rules_multiple_events(0, 1, 0, 0, 0)

        self.assertEqual(len(runner.monitors), 1)
        self.assertIsInstance(runner.monitors[0], WatchdogMonitor)
        self.assertEqual(len(runner.handlers), 1)
        self.assertIsInstance(runner.handlers[0], PapermillHandler)
        self.assertEqual(len(runner.conductors), 1)
        self.assertIsInstance(runner.conductors[0], LocalPythonConductor)

    # Test benchmark patterns setup correctly
    def testPatterns(self)->None:
        jobs = 3
        runner = multiple_rules_multiple_events(jobs, 1, 0, 0, 0)

        patterns = runner.monitors[0].get_patterns()

        self.assertEqual(len(patterns), jobs)

        for j in range(jobs):
            pattern = [p for p in patterns.values()][j]

            self.assertIsInstance(pattern, FileEventPattern)
            self.assertEqual(pattern.name, f"pattern_{j}")
            self.assertEqual(pattern.triggering_path, f"testing/file_{j}.txt")
            self.assertEqual(pattern.triggering_file, "input")
            self.assertEqual(pattern.outputs, {})
            self.assertEqual(pattern.parameters, {})
            self.assertEqual(pattern.recipe, "recipe_one")
            self.assertEqual(pattern.sweep, {})

    # Test benchmark recipes setup correctly
    def testRecipes(self)->None:
        runner = multiple_rules_multiple_events(0, 1, 0, 0, 0)

        recipes = runner.monitors[0].get_recipes()

        self.assertEqual(len(recipes), 1)

        recipe = [r for r in recipes.values()][0]

        self.assertIsInstance(recipe, JupyterNotebookRecipe)
        self.assertEqual(recipe.name, "recipe_one")
        self.assertEqual(recipe.recipe, read_notebook("../notebooks/test.ipynb"))
        self.assertEqual(recipe.requirements, {})
        self.assertEqual(recipe.parameters, {})

    # Test benchmark scheduling for single job
    def testExecutionSmall(self)->None:
        jobs = 1
        _ = multiple_rules_multiple_events(jobs, 1, 0, 0, 0)

        self.assertEqual(len(os.listdir(DEFAULT_JOB_QUEUE_DIR)), jobs)
        for job in os.listdir(DEFAULT_JOB_QUEUE_DIR):
            job_dir = os.path.join(DEFAULT_JOB_QUEUE_DIR, job)

            print(os.listdir(job_dir))

            self.assertTrue(os.path.exists(os.path.join(job_dir, "recipe.ipynb")))
            self.assertTrue(os.path.exists(os.path.join(job_dir, META_FILE)))
            self.assertTrue(os.path.exists(os.path.join(job_dir, JOB_FILE)))

        self.assertEqual(len(os.listdir(DEFAULT_JOB_OUTPUT_DIR)), 0)

    # Test benchmark scheduling for multiple jobs
    def testExecutionLarge(self)->None:
        jobs = 100
        _ = multiple_rules_multiple_events(jobs, 1, 0, 0, 0)

        self.assertEqual(len(os.listdir(DEFAULT_JOB_QUEUE_DIR)), jobs)
        for job in os.listdir(DEFAULT_JOB_QUEUE_DIR):
            job_dir = os.path.join(DEFAULT_JOB_QUEUE_DIR, job)

            self.assertTrue(os.path.exists(os.path.join(job_dir, "recipe.ipynb")))
            self.assertTrue(os.path.exists(os.path.join(job_dir, META_FILE)))
            self.assertTrue(os.path.exists(os.path.join(job_dir, JOB_FILE)))

        self.assertEqual(len(os.listdir(DEFAULT_JOB_OUTPUT_DIR)), 0)
    

class MultipleRulesSingleEventTests(unittest.TestCase):
    def setUp(self)->None:
        super().setUp()
        setup()

    def tearDown(self)->None:
        super().tearDown()
        teardown()

    # Test benchmark runner setup correctly
    def testSetup(self)->None:
        runner = multiple_rules_single_event(0, 1, 0, 0, 0)

        self.assertEqual(len(runner.monitors), 1)
        self.assertIsInstance(runner.monitors[0], WatchdogMonitor)
        self.assertEqual(len(runner.handlers), 1)
        self.assertIsInstance(runner.handlers[0], PapermillHandler)
        self.assertEqual(len(runner.conductors), 1)
        self.assertIsInstance(runner.conductors[0], LocalPythonConductor)

    # Test benchmark patterns setup correctly
    def testPatterns(self)->None:
        jobs = 3
        runner = multiple_rules_single_event(jobs, 1, 0, 0, 0)

        patterns = runner.monitors[0].get_patterns()

        self.assertEqual(len(patterns), jobs)

        for j in range(jobs):
            pattern = [p for p in patterns.values()][j]

            self.assertIsInstance(pattern, FileEventPattern)
            self.assertEqual(pattern.name, f"pattern_{j}")
            self.assertEqual(pattern.triggering_path, f"testing/*")
            self.assertEqual(pattern.triggering_file, "input")
            self.assertEqual(pattern.outputs, {})
            self.assertEqual(pattern.parameters, {})
            self.assertEqual(pattern.recipe, "recipe_one")
            self.assertEqual(pattern.sweep, {})

    # Test benchmark recipes setup correctly
    def testRecipes(self)->None:
        runner = multiple_rules_single_event(0, 1, 0, 0, 0)

        recipes = runner.monitors[0].get_recipes()

        self.assertEqual(len(recipes), 1)

        recipe = [r for r in recipes.values()][0]

        self.assertIsInstance(recipe, JupyterNotebookRecipe)
        self.assertEqual(recipe.name, "recipe_one")
        self.assertEqual(recipe.recipe, read_notebook("../notebooks/test.ipynb"))
        self.assertEqual(recipe.requirements, {})
        self.assertEqual(recipe.parameters, {})

    # Test benchmark scheduling for single job
    def testExecutionSmall(self)->None:
        jobs = 1
        _ = multiple_rules_single_event(jobs, 1, 0, 0, 0)

        self.assertEqual(len(os.listdir(DEFAULT_JOB_QUEUE_DIR)), jobs)
        for job in os.listdir(DEFAULT_JOB_QUEUE_DIR):
            job_dir = os.path.join(DEFAULT_JOB_QUEUE_DIR, job)

            self.assertTrue(os.path.exists(os.path.join(job_dir, "recipe.ipynb")))
            self.assertTrue(os.path.exists(os.path.join(job_dir, META_FILE)))
            self.assertTrue(os.path.exists(os.path.join(job_dir, JOB_FILE)))

        self.assertEqual(len(os.listdir(DEFAULT_JOB_OUTPUT_DIR)), 0)

    # Test benchmark scheduling for multiple jobs
    def testExecutionLarge(self)->None:
        jobs = 100
        _ = multiple_rules_single_event(jobs, 1, 0, 0, 0)

        self.assertEqual(len(os.listdir(DEFAULT_JOB_QUEUE_DIR)), jobs)
        for job in os.listdir(DEFAULT_JOB_QUEUE_DIR):
            job_dir = os.path.join(DEFAULT_JOB_QUEUE_DIR, job)

            self.assertTrue(os.path.exists(os.path.join(job_dir, "recipe.ipynb")))
            self.assertTrue(os.path.exists(os.path.join(job_dir, META_FILE)))
            self.assertTrue(os.path.exists(os.path.join(job_dir, JOB_FILE)))

        self.assertEqual(len(os.listdir(DEFAULT_JOB_OUTPUT_DIR)), 0)
 

class SingleRuleMultipleEventsTests(unittest.TestCase):
    def setUp(self)->None:
        super().setUp()
        setup()

    def tearDown(self)->None:
        super().tearDown()
        teardown()

    # Test benchmark runner setup correctly
    def testSetup(self)->None:
        runner = single_rule_multiple_events(0, 1, 0, 0, 0)

        self.assertEqual(len(runner.monitors), 1)
        self.assertIsInstance(runner.monitors[0], WatchdogMonitor)
        self.assertEqual(len(runner.handlers), 1)
        self.assertIsInstance(runner.handlers[0], PapermillHandler)
        self.assertEqual(len(runner.conductors), 1)
        self.assertIsInstance(runner.conductors[0], LocalPythonConductor)

    # Test benchmark patterns setup correctly
    def testPatterns(self)->None:
        runner = single_rule_multiple_events(0, 1, 0, 0, 0)

        patterns = runner.monitors[0].get_patterns()

        self.assertEqual(len(patterns), 1)

        pattern = [p for p in patterns.values()][0]

        self.assertIsInstance(pattern, FileEventPattern)
        self.assertEqual(pattern.name, f"pattern_one")
        self.assertEqual(pattern.triggering_path, f"testing/*")
        self.assertEqual(pattern.triggering_file, "input")
        self.assertEqual(pattern.outputs, {})
        self.assertEqual(pattern.parameters, {})
        self.assertEqual(pattern.recipe, "recipe_one")
        self.assertEqual(pattern.sweep, {})

    # Test benchmark recipes setup correctly
    def testRecipes(self)->None:
        runner = single_rule_multiple_events(0, 1, 0, 0, 0)

        recipes = runner.monitors[0].get_recipes()

        self.assertEqual(len(recipes), 1)

        recipe = [r for r in recipes.values()][0]

        self.assertIsInstance(recipe, JupyterNotebookRecipe)
        self.assertEqual(recipe.name, "recipe_one")
        self.assertEqual(recipe.recipe, read_notebook("../notebooks/test.ipynb"))
        self.assertEqual(recipe.requirements, {})
        self.assertEqual(recipe.parameters, {})

    # Test benchmark scheduling for single job
    def testExecutionSmall(self)->None:
        jobs = 1
        _ = single_rule_multiple_events(jobs, 1, 0, 0, 0)

        self.assertEqual(len(os.listdir(DEFAULT_JOB_QUEUE_DIR)), jobs)
        for job in os.listdir(DEFAULT_JOB_QUEUE_DIR):
            job_dir = os.path.join(DEFAULT_JOB_QUEUE_DIR, job)

            self.assertTrue(os.path.exists(os.path.join(job_dir, "recipe.ipynb")))
            self.assertTrue(os.path.exists(os.path.join(job_dir, META_FILE)))
            self.assertTrue(os.path.exists(os.path.join(job_dir, JOB_FILE)))

        self.assertEqual(len(os.listdir(DEFAULT_JOB_OUTPUT_DIR)), 0)

    # Test benchmark scheduling for multiple jobs
    def testExecutionLarge(self)->None:
        jobs = 100
        _ = single_rule_multiple_events(jobs, 1, 0, 0, 0)

        self.assertEqual(len(os.listdir(DEFAULT_JOB_QUEUE_DIR)), jobs)
        for job in os.listdir(DEFAULT_JOB_QUEUE_DIR):
            job_dir = os.path.join(DEFAULT_JOB_QUEUE_DIR, job)

            self.assertTrue(os.path.exists(os.path.join(job_dir, "recipe.ipynb")))
            self.assertTrue(os.path.exists(os.path.join(job_dir, META_FILE)))
            self.assertTrue(os.path.exists(os.path.join(job_dir, JOB_FILE)))

        self.assertEqual(len(os.listdir(DEFAULT_JOB_OUTPUT_DIR)), 0)
 

class SingleRuleSingleEventParallelTests(unittest.TestCase):
    def setUp(self)->None:
        super().setUp()
        setup()

    def tearDown(self)->None:
        super().tearDown()
        teardown()

    # Test benchmark runner setup correctly
    def testSetup(self)->None:
        runner = single_rule_single_event_parallel(2, 1, 0, 0, 0)

        self.assertEqual(len(runner.monitors), 1)
        self.assertIsInstance(runner.monitors[0], WatchdogMonitor)
        self.assertEqual(len(runner.handlers), 1)
        self.assertIsInstance(runner.handlers[0], PapermillHandler)
        self.assertEqual(len(runner.conductors), 1)
        self.assertIsInstance(runner.conductors[0], LocalPythonConductor)

    # Test benchmark patterns setup correctly
    def testPatterns(self)->None:
        runner = single_rule_single_event_parallel(2, 1, 0, 0, 0)

        patterns = runner.monitors[0].get_patterns()

        self.assertEqual(len(patterns), 1)

        pattern = [p for p in patterns.values()][0]

        self.assertIsInstance(pattern, FileEventPattern)
        self.assertEqual(pattern.name, f"pattern_one")
        self.assertEqual(pattern.triggering_path, f"testing/*")
        self.assertEqual(pattern.triggering_file, "input")
        self.assertEqual(pattern.outputs, {})
        self.assertEqual(pattern.parameters, {})
        self.assertEqual(pattern.recipe, "recipe_one")
        self.assertEqual(pattern.sweep, {
            "var": {
                SWEEP_START: 1,
                SWEEP_STOP: 2,
                SWEEP_JUMP: 1
            }
        })

    # Test benchmark recipes setup correctly
    def testRecipes(self)->None:
        runner = single_rule_single_event_parallel(2, 1, 0, 0, 0)

        recipes = runner.monitors[0].get_recipes()

        self.assertEqual(len(recipes), 1)

        recipe = [r for r in recipes.values()][0]

        self.assertIsInstance(recipe, JupyterNotebookRecipe)
        self.assertEqual(recipe.name, "recipe_one")
        self.assertEqual(recipe.recipe, read_notebook("../notebooks/test.ipynb"))
        self.assertEqual(recipe.requirements, {})
        self.assertEqual(recipe.parameters, {})

    # Test benchmark scheduling for small job amount
    def testExecutionSmall(self)->None:
        jobs = 2
        _ = single_rule_single_event_parallel(jobs, 1, 0, 0, 0)

        self.assertEqual(len(os.listdir(DEFAULT_JOB_QUEUE_DIR)), jobs)
        for job in os.listdir(DEFAULT_JOB_QUEUE_DIR):
            job_dir = os.path.join(DEFAULT_JOB_QUEUE_DIR, job)

            self.assertTrue(os.path.exists(os.path.join(job_dir, "recipe.ipynb")))
            self.assertTrue(os.path.exists(os.path.join(job_dir, META_FILE)))
            self.assertTrue(os.path.exists(os.path.join(job_dir, JOB_FILE)))

        self.assertEqual(len(os.listdir(DEFAULT_JOB_OUTPUT_DIR)), 0)

    # Test benchmark scheduling for large job amount
    def testExecutionLarge(self)->None:
        jobs = 100
        _ = single_rule_single_event_parallel(jobs, 1, 0, 0, 0)

        self.assertEqual(len(os.listdir(DEFAULT_JOB_QUEUE_DIR)), jobs)
        for job in os.listdir(DEFAULT_JOB_QUEUE_DIR):
            job_dir = os.path.join(DEFAULT_JOB_QUEUE_DIR, job)

            self.assertTrue(os.path.exists(os.path.join(job_dir, "recipe.ipynb")))
            self.assertTrue(os.path.exists(os.path.join(job_dir, META_FILE)))
            self.assertTrue(os.path.exists(os.path.join(job_dir, JOB_FILE)))

        self.assertEqual(len(os.listdir(DEFAULT_JOB_OUTPUT_DIR)), 0)
 

class SingleRuleSingleEventSequentialTests(unittest.TestCase):
    def setUp(self)->None:
        super().setUp()
        setup()

    def tearDown(self)->None:
        super().tearDown()
        teardown()

    # Test benchmark runner setup correctly
    def testSetup(self)->None:
        runner = single_rule_single_event_sequential(0, 1, 0, 0, 0)

        self.assertEqual(len(runner.monitors), 1)
        self.assertIsInstance(runner.monitors[0], WatchdogMonitor)
        self.assertEqual(len(runner.handlers), 1)
        self.assertIsInstance(runner.handlers[0], PapermillHandler)
        self.assertEqual(len(runner.conductors), 1)
        self.assertIsInstance(runner.conductors[0], LocalPythonConductor)

    # Test benchmark patterns setup correctly
    def testPatterns(self)->None:
        runner = single_rule_single_event_sequential(0, 1, 0, 0, 0)

        patterns = runner.monitors[0].get_patterns()

        self.assertEqual(len(patterns), 1)

        pattern = [p for p in patterns.values()][0]

        self.assertIsInstance(pattern, FileEventPattern)
        self.assertEqual(pattern.name, f"pattern_one")
        self.assertEqual(pattern.triggering_path, f"testing/*")
        self.assertEqual(pattern.triggering_file, "INPUT_FILE")
        self.assertEqual(pattern.outputs, {})
        self.assertEqual(pattern.parameters, {
            "MAX_COUNT": 0
        })
        self.assertEqual(pattern.recipe, "recipe_two")
        self.assertEqual(pattern.sweep, {})

    # Test benchmark recipes setup correctly
    def testRecipes(self)->None:
        runner = single_rule_single_event_sequential(0, 1, 0, 0, 0)

        recipes = runner.monitors[0].get_recipes()

        self.assertEqual(len(recipes), 1)

        recipe = [r for r in recipes.values()][0]

        self.assertIsInstance(recipe, JupyterNotebookRecipe)
        self.assertEqual(recipe.name, "recipe_two")
        self.assertEqual(recipe.recipe, read_notebook("../notebooks/sequential.ipynb"))
        self.assertEqual(recipe.requirements, {})
        self.assertEqual(recipe.parameters, {})

    # Test benchmark scheduling for single job
    def testExecutionSmall(self)->None:
        jobs = 1
        _ = single_rule_single_event_sequential(jobs, 1, 0, 0, 0)

        self.assertEqual(len(os.listdir(DEFAULT_JOB_OUTPUT_DIR)), jobs)
        for job in os.listdir(DEFAULT_JOB_OUTPUT_DIR):
            job_dir = os.path.join(DEFAULT_JOB_OUTPUT_DIR, job)

            self.assertTrue(os.path.exists(os.path.join(job_dir, "recipe.ipynb")))
            self.assertTrue(os.path.exists(os.path.join(job_dir, "job.ipynb")))
            self.assertTrue(os.path.exists(os.path.join(job_dir, "result.ipynb")))
            self.assertTrue(os.path.exists(os.path.join(job_dir, META_FILE)))
            self.assertTrue(os.path.exists(os.path.join(job_dir, JOB_FILE)))

        self.assertEqual(len(os.listdir(DEFAULT_JOB_QUEUE_DIR)), 0)

    # Test benchmark scheduling for multiple jobs
    def testExecutionLarge(self)->None:
        jobs = 100
        _ = single_rule_single_event_sequential(jobs, 1, 0, 0, 0)

        self.assertEqual(len(os.listdir(DEFAULT_JOB_OUTPUT_DIR)), jobs)
        for job in os.listdir(DEFAULT_JOB_OUTPUT_DIR):
            job_dir = os.path.join(DEFAULT_JOB_OUTPUT_DIR, job)

            self.assertTrue(os.path.exists(os.path.join(job_dir, "recipe.ipynb")))
            self.assertTrue(os.path.exists(os.path.join(job_dir, "job.ipynb")))
            self.assertTrue(os.path.exists(os.path.join(job_dir, "result.ipynb")))
            self.assertTrue(os.path.exists(os.path.join(job_dir, META_FILE)))
            self.assertTrue(os.path.exists(os.path.join(job_dir, JOB_FILE)))

        self.assertEqual(len(os.listdir(DEFAULT_JOB_QUEUE_DIR)), 0)
 
