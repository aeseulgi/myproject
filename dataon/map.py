import folium
import pandas as pd

class display_map:
    
    def __init__(self):

        self.seoul_center = [37.5065, 127.0360]
        self.seoul_map = folium.Map(location=self.seoul_center, zoom_start=12)
        self.han_river_points = [(37.5448, 127.1131), (37.5266, 127.0645), (37.5147, 126.9964), (37.5167, 126.9581)]
        self.load_data()
        self.display()
    
    def load_data(self):

        river_1018640 = pd.read_csv("./data_predict/1018640_pred.csv")
        river_1018662 = pd.read_csv("./data_predict/1018662_pred.csv")
        river_1018680 = pd.read_csv("./data_predict/1018680_pred.csv")
        river_1018683 = pd.read_csv("./data_predict/1018683_pred.csv")

        river_1018640.drop("Unnamed: 0", axis = 1, inplace = True)
        river_1018662.drop("Unnamed: 0", axis = 1, inplace = True)
        river_1018680.drop("Unnamed: 0", axis = 1, inplace = True)
        river_1018683.drop("Unnamed: 0", axis = 1, inplace = True)
        
        self.water_level_data = {
        "1_hour_later": [round(float(river_1018640.iloc[0]), 2), round(float(river_1018662.iloc[0]), 2), round(float(river_1018680.iloc[0]), 2), round(float(river_1018683.iloc[0]), 2)],
        "2_hours_later": [round(float(river_1018640.iloc[1]), 2), round(float(river_1018662.iloc[1]), 2), round(float(river_1018680.iloc[1]), 2), round(float(river_1018683.iloc[1]), 2)],
        "3_hours_later": [round(float(river_1018640.iloc[2]), 2), round(float(river_1018662.iloc[2]), 2), round(float(river_1018680.iloc[2]), 2), round(float(river_1018683.iloc[2]), 2)],
    }
    
    def display(self):
        bridge_list = ["광진교", "청담대교", "잠수교", "한강대교"]

        for i, point in enumerate(self.han_river_points):
            popup_content = f"<b>{bridge_list[i]}</b><br>"
            for time, levels in self.water_level_data.items():
                popup_content += f"{time}: {levels[i]}m<br>"

            folium.Marker(location=point, popup=popup_content).add_to(self.seoul_map)

        self.seoul_map.save("./MAP/han_river_water_levels.html")
    

    

    