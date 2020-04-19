import os
import shutil

source_dir = '../output'
target_dir = '../output_result'
outputdirs = ['japan', 'feb0416', 'jar0416', 'mar0416']

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
for dir_name in outputdirs:
    outputdir = os.path.join(source_dir, dir_name)
    targetdir = os.path.join(target_dir, dir_name)
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)
    xps = os.listdir(outputdir)
    for xp in xps:
        xp_path = os.path.join(outputdir, xp)
        target_path = os.path.join(targetdir, xp)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        file_names = os.listdir(xp_path)
        for file_name in file_names:
            if file_name == 'config.json' or '.txt' in file_name:
                source_file = os.path.join(xp_path, file_name)
                shutil.copy(source_file, target_path)
