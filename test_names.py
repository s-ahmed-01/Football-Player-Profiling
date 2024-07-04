import unittest
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import unidecode

# Assuming the get_top_5_stats and find_closest_players_cosine functions are defined as provided earlier

# Sample data
mock_data = {
    'Player': ['Lionel Messi', 'Cristiano Ronaldo', 'Neymar', 'Kylian Mbappe', 'Karim Benzema'],
    'Position': ['Forward', 'Forward', 'Forward', 'Forward', 'Forward'],
    'Position_2': [np.nan, np.nan, np.nan, np.nan, np.nan],
    'Stat1': [90, 88, 85, 92, 87],
    'Stat2': [85, 90, 88, 91, 86],
    'Stat3': [80, 82, 84, 89, 83],
    'Stat4': [95, 93, 91, 94, 90],
    'Stat5': [89, 87, 86, 92, 88],
}

df = pd.DataFrame(mock_data)

# Mock latent features (for simplicity, we use random numbers here)
np.random.seed(0)
latent_features = np.random.rand(len(df), 5)

class TestFindClosestPlayersCosine(unittest.TestCase):

    def test_valid_name(self):
        result = find_closest_players_cosine(df, latent_features, "Lionel Messi")
        self.assertIsNotNone(result)
        self.assertFalse(result.empty)
        self.assertIn('Player', result.columns)
    
    def test_invalid_name(self):
        with self.assertRaises(ValueError):
            find_closest_players_cosine(df, latent_features, "Leo Messi")
    
    def test_empty_name(self):
        with self.assertRaises(ValueError):
            find_closest_players_cosine(df, latent_features, "")
    
    def test_case_insensitivity(self):
        result = find_closest_players_cosine(df, latent_features, "lionel messi")
        self.assertIsNotNone(result)
        self.assertFalse(result.empty)
        self.assertIn('Player', result.columns)
    
    def test_name_with_accent(self):
        # Assuming the function can handle "Kylian Mbappe" with or without the accent
        result = find_closest_players_cosine(df, latent_features, "kylian mbappe")
        self.assertIsNotNone(result)
        self.assertFalse(result.empty)
        self.assertIn('Player', result.columns)

if __name__ == '__main__':
    unittest.main()