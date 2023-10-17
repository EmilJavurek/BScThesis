#WATCHOUT !!! DO NOT ACCIDENTALY RUN THIS - creates folders!!!
"""
Inside the folder where this script is run, it will create a "Network_Library" folder,
inside which "p_0" to "p_20" folders are created.
"""

import os
current_dir = os.getcwd()

#create main folder
final_dir = os.path.join(current_dir, r'Network_Library')
if not os.path.exists(final_dir):
    os.makedirs(final_dir)

#create subfolders per p
os.chdir(final_dir)
for index in range(21):
    foldername = f"p_{index}"
    os.mkdir(foldername)
