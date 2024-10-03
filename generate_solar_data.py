import pandas as pd
import numpy as np

def generate_solar_data(samples= 100):
    np.random.seed(32)

    data = {
        'solar_irradiance': np.random.uniform(3.0, 7.0, samples),
        'roof_angle': np.random.uniform(0, 45, samples), #perfectly flat to perfectly 45 degree rooftop
        'roof_azimuth': np.random.uniform(0, 359, samples), #direction of the roof, 0 = true north
        'shade_factor': np.random.uniform(0, 50, samples), #
        'roof_area': np.random.uniform(20, 100, samples), #size of the roof
        'electricity_rate': np.random.uniform(0.08, 0.20, samples),
        'latitude': np.random.uniform(25, 50, samples), #southern texas to southern canada
        'longtitude': np.random.uniform(-125, -65, samples) #east cost to the west coast of the united nations
    }

    df = pd.DataFrame(data)

    df["suitability_score"] = (
        (df["solar_irradiance"] / 7.0) * 0.3 + (1 - df["shade_factor"] / 100) * 0.2 + (df['roof_area'] / 100) * 0.2 + (df['electricity_rate'] / 0.20) * 0.20 + (np.abs(df["roof_angle"] - 30) / 30) * 0.1
    ).clip(0,1)

    return df

solar_data = generate_solar_data()

solar_data.to_csv("solar_data.csv", index = False)

#display
print('Synthetic data has been generated and saved to "solar_data.csv"')
print(solar_data.head())