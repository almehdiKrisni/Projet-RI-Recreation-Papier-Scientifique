import unittest
from utils import sum_two_numbers

class SommeDeuxInputTest(unittest.TestCase):

    def test_should_return_4_when_2_inputs_valued_as_2(self):
        # Given two parameters with value 2
        param1 = 2
        param2 = 2

        # When subject under test sum
        resultat = sum_two_numbers(param1, param2)
        
        # Then result equals 4
        expectedResult = 4
        self.assertEqual(expectedResult, resultat)


    def test_should_return_6_when_2_inputs_valued_as_3(self):
        # Given two parameters with value 3
        param1 = 3
        param2 = 3

        # When subject under test sum
        resultat = sum_two_numbers(param1, param2)
        
        # Then result equals 6
        expectedResult = 6
        self.assertEqual(expectedResult, resultat)


    def test_raise_TypeError_if_at_least_one_param_equals_None(self):
        # Given two parameters with value None
        param1 = None
        param2 = None

        # When subject under test sum
        
        # Should throw an exception
        with self.assertRaises(TypeError):
            sum_two_numbers(param1, param2)




if __name__ == '__main__':
    unittest.main()