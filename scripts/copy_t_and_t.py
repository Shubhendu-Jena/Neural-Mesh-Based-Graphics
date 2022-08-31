import glob
import numpy as np
import shutil
from pathlib import Path

destination_folder = str(Path(__file__).parent.resolve()) + '/../' + 'data/tanks_and_temples'
source_folder = str(Path(__file__).parent.resolve()) + '/../' + 'data/split_meshes' 
folder_list = glob.glob(destination_folder + "/*/", recursive = True)
for subfolder in folder_list:
	subsubfolder_list = glob.glob(subfolder+'*/', recursive = True)
	for subsubfolder in subsubfolder_list:
		folder_name = glob.glob(subsubfolder+'*/', recursive = True)
		fg_mesh_dst = folder_name[0] + 'delaunay_photometric_fg_0.50.ply'
		bg_mesh_dst = folder_name[0] + 'delaunay_photometric_bg_0.50.ply'
		copy_src_fg = source_folder + '/' + folder_name[0].split('/')[-3]  
		copy_src_bg = source_folder + '/' + folder_name[0].split('/')[-3] 
		copy_src_fg = copy_src_fg + '/delaunay_photometric_fg_0.50.ply'
		copy_src_bg = copy_src_bg + '/delaunay_photometric_bg_0.50.ply' 
		shutil.copyfile(copy_src_fg, fg_mesh_dst)
		shutil.copyfile(copy_src_bg, bg_mesh_dst)
		
