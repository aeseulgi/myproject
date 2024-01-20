from loading import aws, lvl
from mstl import dataset_stl
from preprocess import Preprocessing_minmax
from modeling import predict_model
from multiprocessing import Pool
import multiprocessing
from map import display_map


def cleaning():
    print("Preparing data...")                
    AWS = aws()
    LVL = lvl()

    print("Updating data to recent")
    AWS.update_data()
    LVL.update_data()

    for lvl_code in LVL.code_needed:
        for aws_code in LVL.matching_dict[lvl_code]:
            AWS.wind_dir_triangulation(aws_code, LVL.angle_dict[lvl_code])

    print("Imputing missing values")
    AWS.missingInput()
    LVL.missingInput()
    
    AWS.apply_windSpd_windDif()

    print("Saving data...to cleaned version")
    AWS.save_EDAdata()
    LVL.save_EDAdata()
    
def decomposing(code):
    dataset_stl(code)
    return
    
def modeling(code):
    if code not in ["1018640", "1018662", "1018680", "1018683"]: 
        print("Sorry we are not prepared for that location.")
    print("Decomposing the time series data")    
    decomposing(code)
    
    print("Reducing variables by PCA")

    pre = Preprocessing_minmax(code)
    pre.merging()
    pre.do_blocking()
    pre.do_minmax_and_pca()
    
    full_data = pre.pca_full_data
    full_data_test = pre.pca_full_data_test
    full_data_final_test = pre.pca_full_data_final_test
    
    print("Predicting...")
    model = predict_model(full_data, code)
    model.eval_model(pre)

if __name__ == "__main__":
    cleaning() # connection error가 뜨면 반복해서 다시 돌려주세요.
    pool = multiprocessing.Pool()
    pool.map(modeling, ["1018640", "1018662", "1018680", "1018683"])
    # modeling("1018680")    $# if running only single location, cannot display the map: gives an error
    print("Displaying the result on the map")
    m = display_map()
    
    