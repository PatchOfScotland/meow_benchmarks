
import unittest

from meow_base.benchmarking.mrme import multiple_rules_multiple_events

from shared import setup, teardown


class MultipleRulesMultipleEventsTests(unittest.TestCase):
    def setUp(self)->None:
        super().setUp()
        setup()

    def tearDown(self)->None:
        super().tearDown()
        teardown()

    #
    def testSetup(self)->None:
        runner = multiple_rules_multiple_events(0, 0, 0, 0, 0)

    #
    def testPatterns(self)->None:
        pass
    
    #
    def testRecipes(self)->None:
        pass

    #
    def testExecutionSmall(self)->None:
        pass

    #
    def testExecutionLarge(self)->None:
        pass
    
    
