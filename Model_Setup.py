# A AND B IN FRONT OF IMAGES HAVE NO CORRELATIONS TO CLASSES, WERE JUST USED FOR DATA COMBINATION
import os
import pandas as pd
import shutil

# folder path
path = "Series1/Data"
# get all files in the folder
files = os.listdir(path)
# iterate through all files
for index, file in enumerate(files):
    # if the file is a jpg file
    if file.endswith(".jpg"):
        # rename the file by adding a_ to the beginning of the file name
        os.rename(os.path.join(path, file), os.path.join(path, "a_" + file))

# folder path
path = "Series2/Data"
# get all files in the folder
files = os.listdir(path)
# iterate through all files
for index, file in enumerate(files):
    # if the file is a jpg file
    if file.endswith(".jpg"):
        # rename the file by adding b_ to the beginning of the file name
        os.rename(os.path.join(path, file), os.path.join(path, "b_" + file))

dfa = pd.read_csv("Series1/series_1_list_read.csv")
dfb = pd.read_csv("Series2/series_2_list_read.csv")

# in df1, for each row in column image_name, add a_ to the beginning of the value
dfa["Image"] = "a_" + dfa["Image"]
# in df2, for each row in column image_name, add b_ to the beginning of the value
dfb["Image"] = "b_" + dfb["Image"]
# combine the two data frames and put them into a new dataframe in a new directory called Data
df = pd.concat([dfa, dfb], axis=0)
# create a new directory called Data
os.makedirs("Data", exist_ok=True)
df.to_csv("Data/data_directory.csv", index=False)


def combine_jpg_files(series1_dir, series2_dir, combined_dir):
    if not os.path.exists(combined_dir):
        os.makedirs(combined_dir)

    # Function to copy .jpg files from a source directory to a destination directory
    def copy_jpg_files(source_dir, dest_dir):
        for item in os.listdir(source_dir):
            if item.endswith(".jpg"):
                source_path = os.path.join(source_dir, item)
                dest_path = os.path.join(dest_dir, item)
                if os.path.exists(dest_path):
                    base, extension = os.path.splitext(item)
                    counter = 1
                    new_filename = f"{base}_{counter}{extension}"
                    new_dest_path = os.path.join(dest_dir, new_filename)
                    while os.path.exists(new_dest_path):
                        counter += 1
                        new_filename = f"{base}_{counter}{extension}"
                        new_dest_path = os.path.join(dest_dir, new_filename)
                    shutil.copy2(source_path, new_dest_path)
                else:
                    shutil.copy2(source_path, dest_path)

    copy_jpg_files(series1_dir, combined_dir)
    copy_jpg_files(series2_dir, combined_dir)


series1_dir = "Series1/Data"
series2_dir = "Series2/Data"
combined_dir = "Data/images"
combine_jpg_files(series1_dir, series2_dir, combined_dir)
