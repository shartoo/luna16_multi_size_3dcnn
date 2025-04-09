import pandas as pd
import math
import glob
import os

from bs4 import BeautifulSoup
from constant import luna

# mhd  or dicom file information csv file path
mhd_info_csv = luna.MHD_INFO_CSV
# csv columns name of postive nodule
pos_annotation_csv_head = ["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"]
# csv columns name of negative nodule
neg_annotation_csv_head = ["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"]
# root path of every patient's annotation information extracted from xml file
extracted_annotation_info_root_path ='/data/LUNA2016/extracted_annotation_infos/'
#columns name of  all information extracted from xml annotations
all_annotation_csv_head = ["patient_id", "anno_index","servicingRadiologistID", "coord_x", "coord_y", "coord_z", "diameter",
                           "malscore","sphericiy", "margin", "spiculation", "texture", "calcification", "internal_structure", "lobulation", "subtlety"]


def merge_nodule_annotation_csv_to_one(nodule_annotation_csv_list,save_file):
    '''

    :param nodule_annotation_csv_list:
    :param save_file:
    :return:
    '''
    annotattion_list = []
    # add patient id to annotation information csv file
    annotation_head = 'patient_id,'+','.join(pos_annotation_csv_head)
    for csv in nodule_annotation_csv_list:  # csv filename like : 1.3.6.1.4.1.14519.5.2.1.6279.6001.106630482085576298661469304872_annos_pos.csv
        with open(csv,'r') as read_csv:
            contents = read_csv.readlines()
            patient_id = os.path.basename(csv).split("_")[0]  # get patient id
            for line in contents[1:]:                         # skip column head
                annotattion_list.append(patient_id+","+line)
    with open(save_file,'w') as save_csv:
        save_csv.write(annotation_head+"\r\n")
        for line in annotattion_list:
            save_csv.write(line)
    print("csv annotation file merged into file ",save_file)

def read_nodule_annotation_from_xml(xml_path,patient_mhd_path_dict,agreement_threshold=0):
    '''
         read annotaion information xml file path

    :param xml_path:                single xml annotation file
    :param patient_mhd_path_dict:   list of all mhd file paths,xml annotation contains several patient,every patient should get real mhd file path from it
    :param agreement_threshold:     every patient's CT image was marked by multi(4 most) doctor,the least agreement to make final mark
    :return:
    '''
    pos_lines = []
    neg_lines = []
    extended_lines = []
    with open(xml_path, 'r') as xml_file:
        markup = xml_file.read()
    xml = BeautifulSoup(markup, features="xml")
    if xml.LidcReadMessage is None:
        return None, None, None
    patient_id = xml.LidcReadMessage.ResponseHeader.SeriesInstanceUid.text
    src_path = None
    if patient_id in patient_mhd_path_dict:
        src_path = patient_mhd_path_dict[patient_id]
    if src_path is None:
        return None, None, None

    print(patient_id)
    mhd_info_pd = pd.read_csv(mhd_info_csv)
    mhd_info_row = mhd_info_pd[mhd_info_pd['patient_id']==patient_id]
    print("information about this patient is:\t",mhd_info_row)
    num_z, height, width = list(mhd_info_row['shape_2'])[0],list(mhd_info_row['shape_1'])[0],list(mhd_info_row['shape_0'])[0]
    print("num_z,height,width are:\t",num_z, height, width)
    origin_x,origin_y,origin_z = list(mhd_info_row['origin_x'])[0],list(mhd_info_row['origin_y'])[0],list(mhd_info_row['origin_z'])[0]
    spacing_x,spacing_y,spacing_z = list(mhd_info_row['spacing_x'])[0],list(mhd_info_row['spacing_y'])[0],list(mhd_info_row['spacing_z'])[0]

    #  a reading session consists of the results consists of a set of markings done by a single
    # reader at a single phase (for these xml files, the unblinded reading phase).
    reading_sessions = xml.LidcReadMessage.find_all("readingSession")

    for reading_session in reading_sessions:
        # print("Sesion")
        servicingRadiologistID = reading_session.servicingRadiologistID.text
        nodules = reading_session.find_all("unblindedReadNodule")
        for nodule in nodules:
            nodule_id = nodule.noduleID.text
            # print("  ", nodule.noduleID)
            rois = nodule.find_all("roi")
            x_min = y_min = z_min = 999999
            x_max = y_max = z_max = -999999
            if len(rois) < 2:
                continue

            for roi in rois:
                z_pos = float(roi.imageZposition.text)
                z_min = min(z_min, z_pos)
                z_max = max(z_max, z_pos)
                edge_maps = roi.find_all("edgeMap")
                for edge_map in edge_maps:
                    x = int(edge_map.xCoord.text)
                    y = int(edge_map.yCoord.text)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                if x_max == x_min:
                    continue
                if y_max == y_min:
                    continue

            x_diameter = x_max - x_min
            x_center = x_min + x_diameter / 2
            y_diameter = y_max - y_min
            y_center = y_min + y_diameter / 2
            z_diameter = z_max - z_min
            z_center = z_min + z_diameter / 2
            z_center -= origin_z
            z_center /= spacing_z

            x_center_perc = round(x_center / width, 4)
            y_center_perc = round(y_center / height, 4)
            z_center_perc = round(z_center / num_z, 4)
            diameter = max(x_diameter , y_diameter)
            diameter_perc = round(max(x_diameter / width, y_diameter / height), 4)

            if nodule.characteristics is None:
                print("!!!!Nodule:", nodule_id, " has no charecteristics")
                continue
            if nodule.characteristics.malignancy is None:
                print("!!!!Nodule:", nodule_id, " has no malignacy")
                continue
            print("nodule in load xml",x_center_perc,y_center_perc,z_center_perc)
            malignacy = nodule.characteristics.malignancy.text
            sphericiy = nodule.characteristics.sphericity.text
            margin = nodule.characteristics.margin.text
            spiculation = nodule.characteristics.spiculation.text
            texture = nodule.characteristics.texture.text
            calcification = nodule.characteristics.calcification.text
            internal_structure = nodule.characteristics.internalStructure.text
            lobulation = nodule.characteristics.lobulation.text
            subtlety = nodule.characteristics.subtlety.text

            line = [nodule_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, malignacy]
            extended_line = [patient_id, nodule_id, servicingRadiologistID,x_center_perc, y_center_perc, z_center_perc,
                             diameter_perc, malignacy,sphericiy, margin, spiculation, texture, calcification, internal_structure, lobulation, subtlety]

            pos_lines.append(line)
            extended_lines.append(extended_line)

        nonNodules = reading_session.find_all("nonNodule")
        for nonNodule in nonNodules:
            z_center = float(nonNodule.imageZposition.text)
            z_center -= origin_z
            z_center /= spacing_z
            x_center = int(nonNodule.locus.xCoord.text)
            y_center = int(nonNodule.locus.yCoord.text)
            nodule_id = nonNodule.nonNoduleID.text
            x_center_perc = round(x_center / width, 4)
            y_center_perc = round(y_center / height, 4)
            z_center_perc = round(z_center / num_z, 4)
            diameter_perc = round(max(6 / width, 6 / height), 4)
            # print("Non nodule!", z_center)
            line = [nodule_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, 0]
            neg_lines.append(line)

    if agreement_threshold > 1:
        filtered_lines = []
        for pos_line1 in pos_lines:
            id1 = pos_line1[0]
            x1 = pos_line1[1]
            y1 = pos_line1[2]
            z1 = pos_line1[3]
            d1 = pos_line1[4]
            overlaps = 0
            for pos_line2 in pos_lines:
                id2 = pos_line2[0]
                if id1 == id2:
                    continue
                x2 = pos_line2[1]
                y2 = pos_line2[2]
                z2 = pos_line2[3]
                d2 = pos_line1[4]
                dist = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2) + math.pow(z1 - z2, 2))
                if dist < d1 or dist < d2:
                    overlaps += 1
            if overlaps >= agreement_threshold:
                filtered_lines.append(pos_line1)
            # else:
            #     print("Too few overlaps")
        pos_lines = filtered_lines

    df_annos = pd.DataFrame(pos_lines, columns=pos_annotation_csv_head)
    df_annos.to_csv(extracted_annotation_info_root_path + patient_id + "_annos_pos_lidc.csv", index=False)
    df_neg_annos = pd.DataFrame(neg_lines, columns=neg_annotation_csv_head)
    df_neg_annos.to_csv(extracted_annotation_info_root_path + patient_id + "_annos_neg_lidc.csv", index=False)

    return pos_lines, neg_lines, extended_lines



def process_lidc_annotations(xml_annotation_like,patient_mhd_path_dict,mhd_all_info_save_path,agreement_threshold=0):
    '''
        extract xml annotation information from xml path

    :param xml_annotation_like:             used for glob,a file path string
    :param patient_mhd_path_dict:           key is patient id,value is the mhd file path
    :param mhd_all_info_save_path:          where to  save the extracted mhd information csv file(not mhd_info_csv)
    :param agreement_threshold:             every nodule was annotated by multi-doctor,the least number
    :return:
    '''
    file_no = 0
    pos_count = 0
    neg_count = 0
    all_lines = []

    xml_paths = glob.glob(xml_annotation_like)
    for xml_path in xml_paths:
        pos, neg, extended = read_nodule_annotation_from_xml(xml_path, patient_mhd_path_dict,agreement_threshold=agreement_threshold)
        if pos is not None:
            pos_count += len(pos)
            neg_count += len(neg)
            file_no += 1
            all_lines += extended
    df_annos = pd.DataFrame(all_lines, columns= all_annotation_csv_head)
    df_annos.to_csv(mhd_all_info_save_path, index=False)

def extract_lidc_every_z_annotations(xml_like,every_z_save_csv,patient_mhd_path_dict):
    xml_paths = glob.glob(xml_like)
    with open(every_z_save_csv,"w") as anno_save:
        anno_save.write("seriesuid,coord_percent_x, coord_percent_y,coord_mm_z,percent_diamater,mm_x,mm_y,diameter_mm,malscore")
        anno_save.write("\r\n")
        for xml_path in xml_paths:
            extended = extract_every_z_from_lidc_xml(xml_path, patient_mhd_path_dict)
            if extended is not None:
                anno_save.write(str(extended).replace(")","").replace("(","").replace("\'","")+"\r\n")


def extract_every_z_from_lidc_xml(xml_path,patient_mhd_path_dict):
    '''
        extract every nodule(ROIs) from xml file,the above method `read_nodule_annotation_from_xml` extract center z coordinates of nodule

        this method was used by UNet to produce more nodule mask

    :param xml_path:                xml file of nodule annotation
    :param patient_mhd_path_dict:   key is patient id,value is its full mhd file path
    :return:                        list of every nodule's coordinates
    '''
    extended_lines = []
    with open(xml_path, 'r') as xml_file:
        markup = xml_file.read()
    xml = BeautifulSoup(markup, features="xml")
    if xml.LidcReadMessage is None:
        return None, None, None
    patient_id = xml.LidcReadMessage.ResponseHeader.SeriesInstanceUid.text

    print("patient id is:\t", patient_id)
    if patient_id in patient_mhd_path_dict:
        src_path = patient_mhd_path_dict[patient_id]
    else:
        return None, None, None

    print(patient_id)
    mhd_info_pd = pd.read_csv(mhd_info_csv)
    mhd_info_row = mhd_info_pd[mhd_info_pd['patient_id'] == patient_id]
    print("information about this patient is:\t", mhd_info_row)
    num_z, height, width = list(mhd_info_row['shape_2'])[0], list(mhd_info_row['shape_1'])[0], \
                           list(mhd_info_row['shape_0'])[0]
    print("num_z,height,width are:\t", num_z, height, width)
    origin_x, origin_y, origin_z = list(mhd_info_row['origin_x'])[0], list(mhd_info_row['origin_y'])[0], \
                                   list(mhd_info_row['origin_z'])[0]
    spacing_x, spacing_y, spacing_z = list(mhd_info_row['spacing_x'])[0], list(mhd_info_row['spacing_y'])[0], \
                                      list(mhd_info_row['spacing_z'])[0]

    #  a reading session consists of the results consists of a set of markings done by a single
    # reader at a single phase (for these xml files, the unblinded reading phase).
    reading_sessions = xml.LidcReadMessage.find_all("readingSession")

    for reading_session in reading_sessions:
        # print("Sesion")
        servicingRadiologistID = reading_session.servicingRadiologistID.text
        nodules = reading_session.find_all("unblindedReadNodule")
        for nodule in nodules:
            nodule_id = nodule.noduleID.text
            if nodule.characteristics is None:
                print("!!!!Nodule:", nodule_id, " has no charecteristics")
                continue
            if nodule.characteristics.malignancy is None:
                print("!!!!Nodule:", nodule_id, " has no malignacy")
                continue
            malignacy = nodule.characteristics.malignancy.text

            rois = nodule.find_all("roi")
            x_min = y_min = z_min = 999999
            x_max = y_max = z_max = -999999
            if len(rois) < 2:
                continue

            for roi in rois:
                z_pos = float(roi.imageZposition.text)
                edge_maps = roi.find_all("edgeMap")
                for edge_map in edge_maps:
                    x = int(edge_map.xCoord.text)
                    y = int(edge_map.yCoord.text)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                if x_max == x_min:
                    continue
                if y_max == y_min:
                    continue

                x_diameter = x_max - x_min
                x_center = x_min + x_diameter / 2
                y_diameter = y_max - y_min
                y_center = y_min + y_diameter / 2

                x_center_perc = round(x_center / width, 4)
                y_center_perc = round(y_center / height, 4)
                diameter_mm = max(x_diameter, y_diameter)
                diameter_perc = round(max(x_diameter / width, y_diameter / height), 4)

                extended_line = patient_id+","+str(round(x_center_perc,4))+","+str(round(y_center_perc,4))+","+str(z_pos)+","+\
                                 str(round(diameter_perc,4))+","+str(x_center),str(y_center)+","+str(diameter_mm)+"," +malignacy
                extended_lines.append(extended_line)

    return extended_lines


