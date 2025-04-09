# -*- coding:utf-8 -*-
'''
 a script to process result produced by lidc_annotation_process.py

 mainly focus on process mm coordinates of annotations
'''

import math
import os
import pandas as pd
from constant import luna
from util import mhd_util,image_util, cube


def draw_percent_cube_by_csv(percent_csv,mhd_info_csv,cube_save_path):
    '''
       coordinate from xml are percent of image shape,
    :param percent_coordinate_csv:
    :return:
    '''
    mhd_pandas_index = mhd_util.read_csv_to_pandas(mhd_info_csv,',')
    percent_pandas = mhd_util.read_csv_to_pandas(percent_csv,',')

    for index, row in percent_pandas.iterrows():
        patient_id = row['patient_id']
        coord_x = float(row['coord_x'])
        coord_y = float(row['coord_y'])
        coord_z = float(row['coord_z'])
        malscore = row['malscore']
        diameter = float(row['diameter'])
        print("read cube coordx,coordy,coordz,diamater,malscore",coord_x,coord_y,coord_z,diameter,malscore)
        png_path = luna.LUNA_EXTRACTED_IMG + '/' + patient_id
        if os.path.exists(png_path):
            cube.draw_percent_cube(png_path, mhd_pandas_index, coord_x, coord_y,
                                   coord_z, diameter, cube_save_path, probility =malscore)
        else:
            print("one patient not exists..", png_path)


def percent_coordinatecsv_to_mmcsv(percent_csv,mhd_info_csv,mmcsv_save):
    '''
       transform percent coordinates to mm coordinates and save into new csv file

    :param percent_csv:     csv file contains all percent coordinates
    :param mmcsv_save:      csv file to save transformed coordinates
    :return:
    '''
    with open(percent_csv,'r') as percent_read:
        head = percent_read.readline()
        print(str(head))
        head = str(head).replace("coord_x,coord_y,coord_z","coord_x,coord_y,coord_z,mm_x,mm_y,mm_z")
        extend_mm_coordinate_content = []
        lines = percent_read.readlines()
        for line in lines:
            cols = line.split(",")
            patient_id = cols[0]
            p_x,p_y,p_z = cols[3],cols[4],cols[5] #row['coord_x'],row['coord_y'],row['coord_z']
            print("one line\t",line)
            print("patient id = ",patient_id)
            print("coordinates is:\t ",p_x,p_y,p_z)
            mm_x,mm_y,mm_z = percent_coordinate_to_mm(patient_id,float(p_x),float(p_y),float(p_z),mhd_info_csv)
            print("after transfered ..:\t",mm_x,mm_y,mm_z)
            line = line.replace(str(p_x)+","+str(p_y)+","+str(p_z),
                                str(p_x)+","+str(p_y)+","+str(p_z)+","+str(mm_x)+","+str(mm_y)+","+str(mm_z))
            print(line)
            extend_mm_coordinate_content.append(line)

        with open(mmcsv_save,"w") as mm:
            mm.write(head)
            for line in extend_mm_coordinate_content:
                mm.write(str(line))
    print("transformed mm coordinates finished..")


def avg_coordinates(csv,threshold,csv_save):
    '''
        add average coordinates to source.
     before:
        patient_id0, coord_x0,coord_y0,coord_z0
        patient_id0, coord_x1,coord_y1,coord_z1
     after:
        patient_id0, coord_x0,coord_y0,coord_z0,avg_x,avg_y,avg_z
        patient_id0, coord_x1,coord_y1,coord_z1,avg_x,avg_y,avg_z
    :param csv:
    :param csv_save:        csv file to save transformed content
    :return:
    '''
    new_content = []
    patient_coords = {}
    with open(csv,'r') as csv_read:
        head = csv_read.readline()
        lines = csv_read.readlines()
        for line in lines:
            cols = line.split(",")
            patient_id = cols[0]
            mm_x,mm_y,mm_z,diam,mals = cols[6],cols[7],cols[8],cols[9],cols[10]
            if patient_id in patient_coords:
                patient_coords[patient_id].append([mm_x,mm_y,mm_z,diam,mals])
            else:
                patient_coords[patient_id]= [[mm_x, mm_y, mm_z, diam,mals]]

    # get average coords
    avg_same_coords = {}

    for key,value in patient_coords.items():
        patient_id = key
        coords = value
        xyzs = []
        for coor in coords: # list of lists
            x,y,z = coor[0],coor[1],coor[2]
            xyzs.append([float(x),float(y),float(z)])

        # find every coordinate's neighbor coordinates (distance smaller than threshold)
        same_coords = {}
        coord_num = len(xyzs)
        i = 0
        while i <coord_num:
            current_x,current_y,current_z =xyzs[i][0],xyzs[i][1],xyzs[i][2]
            same_with_current = str(current_x)+","+str(current_y)+","+str(current_z)
            same_coords[same_with_current] = [[current_x,current_y,current_z]]   # put itself into its neighbor
            j = 0
            while j < coord_num:
                x,y,z = xyzs[j][0],xyzs[j][1],xyzs[j][2]
                dis = math.sqrt((x-current_x)**2+(y-current_y)**2+(z-current_z)**2)            # distance of two coordinates
                if dis< threshold and [x,y,z] not in same_coords[same_with_current]:
                    same_coords[same_with_current].append([x,y,z])
                j = j+1
            i+=1

        # get average of coordinates
        for key,value in same_coords.items():
            cur_x,curr_y,curr_z = key.split(",")
            x,y,z = 0,0,0
            #print("key:  value: ",key,":\t",value)
            if len(value)>0:
                for same_cor in value:
                    #print(same_cor)
                    x = x + same_cor[0]
                    y = y + same_cor[1]
                    z = z + same_cor[2]
                x = round(x /len(value),2)
                y = round(y /len(value),2)
                z = round(z /len(value),2)
                # update dict with average
            avg_same_coords[key] = [x,y,z]

    with open(csv,'r') as csv_read:
        head = csv_read.readline()
        head = head.replace("mm_x,mm_y,mm_z","mm_x,mm_y,mm_z,avg_x,avg_y,avg_z")
        new_content.append(head)
        lines = csv_read.readlines()
        for line in lines:
            cols = line.split(",")
            patient_id = cols[0]
            mm_x,mm_y,mm_z,diam,mals = cols[6],cols[7],cols[8],cols[9],cols[10]
            avg_xyz = avg_same_coords[mm_x+","+mm_y+","+mm_z]
            avg_x,avg_y,avg_z = str(avg_xyz[0]),str(avg_xyz[1]),str(avg_xyz[2])
            new_content.append(line.replace(mm_x+","+mm_y+","+mm_z,
                                           mm_x + "," + mm_y + "," + mm_z+","+avg_x+","+avg_y+","+avg_z))

    with open(csv_save,'w') as info:
        for line in new_content:
            print("line from lidc_coordinate:\t",line)
            info.write(line)

    print("write attachement information to %s finished.."%csv_save)


def add_final_mals(csv,with_real_malsclabel_csv):
    """
            compute real malignancy of every patient. every nodule was labeled by several different readers,
        this step is comfirming a final malignancy label
    :param csv:                         csv file of all mhd information
    :param with_real_malsclabel_csv:    csv file after add real malscore columns
    :return:
    """
    nodule_mals = {}
    with open(csv,'r') as read_csv:
        head = read_csv.readline()
        lines = read_csv.readlines()
        for line in lines:
            cols = line.split(",")
            patient_id = cols[0]
            avg_x,avg_y,avg_z,mals = cols[9],cols[10],cols[11],cols[13]
            key = patient_id+","+avg_x+","+avg_y+","+avg_z
            if key not in nodule_mals:
                nodule_mals[key] = [int(mals)]
            else:
                nodule_mals[key].append(int(mals))

    # compute the real malignancy label
    for key,val in nodule_mals.items():
        mals = val
        print("patient_id and all malscore is:\t",key+"\t:",val)
        non_cancer = 0
        unknow = 0
        cancer = 0
        UNK ="unknow"
        for mal in mals:
            if mal<3:
                non_cancer +=1
            elif mal ==3:
                unknow +=1
            elif mal>3:
                cancer+=1
        real_mal = ""
        if unknow == len(mals):         # all are unknow
            real_mal = UNK
        elif non_cancer/(non_cancer+cancer)>0.5:
            real_mal = "0"
        elif non_cancer/(non_cancer+cancer)==0.5:
            real_mal =UNK
        elif non_cancer/(non_cancer+cancer)<0.5:
            real_mal = "1"

        if real_mal=="0" and unknow>non_cancer:
            real_mal = UNK

        print("non_cancer,unk,cancer,real label", non_cancer, unknow, cancer,real_mal)
        # update the mal label
        nodule_mals[key] = real_mal

    # add real mal columns into csv file
    with_real_mals_content = []
    with open(csv,'r') as read_csv:
        head = read_csv.readline()
        head = head.replace("avg_x,avg_y,avg_z","avg_x,avg_y,avg_z,real_mal")
        with_real_mals_content.append(head)
        lines = read_csv.readlines()
        for line in lines:
            cols = line.split(",")
            patient_id = cols[0]
            avg_x,avg_y,avg_z,mals = cols[9],cols[10],cols[11],cols[13]
            key = patient_id+","+avg_x+","+avg_y+","+avg_z
            real_mal = nodule_mals[key]
            # average coordinates equal to source coordinates
            if avg_x + "," + avg_y +"," + avg_z+","+avg_x + "," + avg_y +"," + avg_z in line:
                with_real_mals_content.append(line.replace(avg_x + "," + avg_y +"," + avg_z + "," + avg_x + "," + avg_y +"," + avg_z,
                                                           avg_x + "," + avg_y + "," + avg_z +","+ avg_x + "," + avg_y + "," + avg_z +","+ real_mal))
            else:
                with_real_mals_content.append(line.replace(avg_x + "," + avg_y +"," + avg_z,
                                                       avg_x + "," + avg_y + "," + avg_z + ","+ real_mal))

    # write the final result with real malscore columns into file
    with open(with_real_malsclabel_csv) as with_mal:
        for line in with_real_mals_content:
            with_mal.write(line)

    print("write result with real malscore columns finished..")



def percent_coordinate_to_mm(patient_id,p_x,p_y,p_z,mhd_info_csv):
    """
      transform percent coordinate to mm coordinates

    :param patient_id:       patient id,used for mapping information from mhd_info_csv
    :param p_x:              x percent coordinate
    :param p_y:
    :param p_z:
    :param mhd_info_csv:    a csv file contains  all mhd information ,such as shape,spacing,origion
    :return:                transformed mm coordinate x,y,z
    """
    png_path = png_path = luna.LUNA_EXTRACTED_IMG + '/'+patient_id
    patient_img = image_util.load_patient_images(png_path, "*_i.png", [])
    mhd_pandas_index = mhd_util.read_csv_to_pandas(mhd_info_csv,',')
    patient_mhd_info = mhd_pandas_index.loc[patient_id]

    z = int(p_z * patient_img.shape[0])
    y = int(p_y * patient_img.shape[1])
    x = int(p_x * patient_img.shape[2])
    orgin_x = float(patient_mhd_info['origin_x'].strip())
    orgin_y = float(patient_mhd_info['origin_y'].strip())
    orgin_z = float(patient_mhd_info['origin_z'].strip())

    right_x = x + orgin_x
    right_y = y + orgin_y
    right_z = z + orgin_z

    return round(right_x,2),round(right_y,2),round(right_z,2)

def draw_all_confirmed_cubes(mm_coordinates_csv,mhd_info_csv,extract_png_path,save_path):
    """
        draw all annotated nodule by luna2016 official
    :param mm_coordinates_csv:
    :param mhd_info_csv:
    :param extract_png_path:
    :param save_path:
    :return:
    """
    coordinates = pd.read_csv(mm_coordinates_csv)
    count = 0
    mhd_info = mhd_util.read_csv_to_pandas(mhd_info_csv)
    for df_index, df_row in coordinates.iterrows():
        patient_id = df_row['seriesuid']
        patient_png_path = os.path.join(extract_png_path,patient_id)
        mm_x = df_row['coordX']
        mm_y = df_row['coordY']
        mm_z = df_row['coordZ']
        diameter = df_row['diameter_mm']
        if os.path.exists(patient_png_path):
            cube.draw_percent_cube(patient_png_path, mm_x, mm_y, mm_z, diameter, save_path, mhd_pandas_index =mhd_info)
        else:
            count +=1
    print("draw all cubes finished...%d cubes missed"%count)

