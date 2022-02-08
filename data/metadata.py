import os


categorical_variables = ['State_Factor', 'building_class']
incomplete_variables = ['year_built', 'energy_star_rating', 'direction_max_wind_speed',	'direction_peak_wind_speed',
                        'max_wind_speed', 'days_with_fog']
competition_name = 'widsdatathon2022'
target_var = 'site_eui'
data_path = './data'
train_data_path = os.path.join(data_path, 'train.csv')
test_data_path = os.path.join(data_path, 'test.csv')
trained_model = {}
