from service import settings_jjyang
import cv2
import pandas
import os
import glob
import numpy
from keras import backend as K
from keras.models import model_from_json

from util.cube import save_cube_img, get_cube_from_img
from util.image.processing import normalize_hu_values
from util.dicom_util import get_pixels_hu, extract_dicom_images_patient, load_dicom_slices
from util.image_util import prepare_image_for_net3D, load_patient_images, rescale_patient_images
from util.ml.metrics import get_3d_pixel_l2_distance
from util.progress_watch import Stopwatch

K.set_image_data_format("channels_last")  # 更新为TF2方式设置
import tensorflow as tf

# 在TF2中设置GPU内存使用方式
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 将GPU内存使用限制为可用内存的30%
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 3)]  # 约30%的10GB显存
            )
    except RuntimeError as e:
        print(e)

CUBE_SIZE = 32
MEAN_PIXEL_VALUE = settings_jjyang.MEAN_PIXEL_VALUE_NODULE
NEGS_PER_POS = 20
P_TH = 0.7

PREDICT_STEP = 12
USE_DROPOUT = False

BOX_size = 20
BOX_depth = 9
# NODULE_CHANCE = 0.5
NODULE_DIAMM = 1.0
CUBE_IMGTYPE_SRC = "_i"

def filter_patient_nodules_predictions(df_nodule_predictions: pandas.DataFrame, patient_id, view_size, png_path):
    patient_mask = load_patient_images(png_path+"/png/", "*_m.png")
    delete_indices = []
    for index, row in df_nodule_predictions.iterrows():
        z_perc = row["coord_z"]
        y_perc = row["coord_y"]
        center_x = int(round(row["coord_x"] * patient_mask.shape[2]))
        center_y = int(round(y_perc * patient_mask.shape[1]))
        center_z = int(round(z_perc * patient_mask.shape[0]))

        mal_score = row["diameter_mm"]
        start_y = center_y - view_size / 2
        start_x = center_x - view_size / 2
        nodule_in_mask = False
        for z_index in [-1, 0, 1]:
            img = patient_mask[z_index + center_z]
            start_x = int(start_x)
            start_y = int(start_y)
            view_size = int(view_size)
            img_roi = img[start_y:start_y + view_size, start_x:start_x + view_size]
            if img_roi.sum() > 255:  # more than 1 pixel of mask.
                nodule_in_mask = True

        if not nodule_in_mask:
            print("Nodule not in mask: ", (center_x, center_y, center_z))
            if mal_score > 0:
                mal_score *= -1
            df_nodule_predictions.loc[index, "diameter_mm"] = mal_score
        else:
            if center_z < 30:
                print("Z < 30: ", patient_id, " center z:", center_z, " y_perc: ", y_perc)
                if mal_score > 0:
                    mal_score *= -1
                df_nodule_predictions.loc[index, "diameter_mm"] = mal_score

            if (z_perc > 0.75 or z_perc < 0.25) and y_perc > 0.85:
                print("SUSPICIOUS FALSEPOSITIVE: ", patient_id, " center z:", center_z, " y_perc: ", y_perc)

            if center_z < 50 and y_perc < 0.30:
                print("SUSPICIOUS FALSEPOSITIVE OUT OF RANGE: ", patient_id, " center z:", center_z, " y_perc: ",
                      y_perc)

    df_nodule_predictions.drop(df_nodule_predictions.index[delete_indices], inplace=True)
    return df_nodule_predictions


def predict_cubes(png_path, model_path, only_patient_id=None,  magnification=1, flip=False):
    patient_id = only_patient_id
    all_predictions_csv = []
    sw = helpers.Stopwatch.start_new()
    json_file = open('../service/workdir/model_loc.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(model_path)
    patient_img = load_patient_images(png_path+"/png/", "*_i.png", [])
    if magnification != 1:
        patient_img = rescale_patient_images(patient_img, (1, 1, 1), magnification)

    patient_mask = load_patient_images(png_path+"/png/", "*_m.png", [])
    if magnification != 1:
        patient_mask = rescale_patient_images(patient_mask, (1, 1, 1), magnification, is_mask_image=True)

    step = PREDICT_STEP
    CROP_SIZE = CUBE_SIZE

    predict_volume_shape_list = [0, 0, 0]
    for dim in range(3):
        dim_indent = 0
        while dim_indent + CROP_SIZE < patient_img.shape[dim]:
            predict_volume_shape_list[dim] += 1
            dim_indent += step

    predict_volume_shape = (predict_volume_shape_list[0], predict_volume_shape_list[1], predict_volume_shape_list[2])
    predict_volume = numpy.zeros(shape=predict_volume_shape, dtype=float)
    print("Predict volume shape: ", predict_volume.shape)
    done_count = 0
    skipped_count = 0
    batch_size = 32
    batch_list = []
    batch_list_coords = []
    patient_predictions_csv = []
    cube_img = None
    annotation_index = 0
    for z in range(0, predict_volume_shape[0]):
        for y in range(0, predict_volume_shape[1]):
            for x in range(0, predict_volume_shape[2]):
                # if cube_img is None:
                cube_img = patient_img[z * step:z * step + CROP_SIZE, y * step:y * step + CROP_SIZE,
                           x * step:x * step + CROP_SIZE]
                cube_mask = patient_mask[z * step:z * step + CROP_SIZE, y * step:y * step + CROP_SIZE,
                            x * step:x * step + CROP_SIZE]

                if cube_mask.sum() < 2000:
                    skipped_count += 1
                else:
                    if flip:
                        cube_img = cube_img[:, :, ::-1]

                    img_prep = prepare_image_for_net3D(cube_img)
                    batch_list.append(img_prep)
                    batch_list_coords.append((z, y, x))
                    if len(batch_list) % batch_size == 0:
                        batch_data = numpy.vstack(batch_list)
                        p = model.predict(batch_data, batch_size=batch_size)
                        for i in range(len(p[0])):
                            p_z = batch_list_coords[i][0]
                            p_y = batch_list_coords[i][1]
                            p_x = batch_list_coords[i][2]
                            nodule_chance = p[0][i][0]
                            predict_volume[p_z, p_y, p_x] = nodule_chance
                            if nodule_chance > P_TH:
                                p_z = p_z * step + CROP_SIZE / 2
                                p_y = p_y * step + CROP_SIZE / 2
                                p_x = p_x * step + CROP_SIZE / 2

                                p_z_perc = round(p_z / patient_img.shape[0], 4)
                                p_y_perc = round(p_y / patient_img.shape[1], 4)
                                p_x_perc = round(p_x / patient_img.shape[2], 4)
                                diameter_mm = round(p[1][i][0], 4)
                                diameter_perc = round(diameter_mm / patient_img.shape[2], 4)
                                nodule_chance = round(nodule_chance, 4)
                                patient_predictions_csv_line = [annotation_index, p_x_perc, p_y_perc, p_z_perc,
                                                                diameter_perc, nodule_chance, diameter_mm]
                                patient_predictions_csv.append(patient_predictions_csv_line)
                                all_predictions_csv.append([patient_id] + patient_predictions_csv_line)
                                annotation_index += 1

                        batch_list = []
                        batch_list_coords = []
                done_count += 1
                if done_count % 10000 == 0:
                    print("Scan: ", done_count, " skipped:", skipped_count)

    df = pandas.DataFrame(patient_predictions_csv,
                          columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "nodule_chance",
                                   "diameter_mm"])
    filter_patient_nodules_predictions(df, patient_id, CROP_SIZE * magnification, png_path)
    # df.to_csv(settings_jjyang.BASE_DIR_SSD+"temp_dir/" + patient_id + ".csv", index=False)
    print(predict_volume.mean())
    print("GPU costs : ", sw.get_elapsed_seconds(), " seconds")
    return df

def reduce_predicts_same_slice(pred_nodules_df):
    rows_filter = []
    pred_nodules_df_local = pred_nodules_df.sort_values(["coord_z", "diameter_mm"], ascending=False)
    if len(pred_nodules_df_local) <= 1:
        return pred_nodules_df_local
    i = 0
    compare_row = pred_nodules_df_local.iloc[0]
    for row_index, row in pred_nodules_df_local[1:].iterrows():
        if compare_row["coord_z"] == row["coord_z"]:
            dist = get_3d_pixel_l2_distance(compare_row, row)
            if dist > 0.2:
                rows_filter.append(row)
        else:
            rows_filter.append(compare_row)
            compare_row = row
        i += 1
    if len(rows_filter) == 0:
        rows_filter.append(compare_row)
    last_row = rows_filter[len(rows_filter)-1]
    if last_row["coord_z"] != compare_row["coord_z"]:
        rows_filter.append(compare_row)
    columns = ["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "nodule_chance", "diameter_mm"]
    res_df = pandas.DataFrame(rows_filter, columns=columns)
    return res_df


def multipule_test(workspace, only_patient_id, CONTINUE_JOB):
    temp_df = []
    for model_version in ["model_loc.hd5", "model_loc_val_0.96.hd5"]:
        print("gpu begin:")
        pred_nodules_df = predict_cubes(workspace, "models/" + model_version, CONTINUE_JOB, only_patient_id=only_patient_id,
                                        magnification=1, flip=False, train_data=True, holdout_no=None,
                                        ext_name="luna16_fs")
        pred_nodules_df = pred_nodules_df[pred_nodules_df["nodule_chance"] > P_TH]
        pred_nodules_df = pred_nodules_df[pred_nodules_df["diameter_mm"] > NODULE_DIAMM]

        temp_df.append(pred_nodules_df)
    temp_dataframe = pandas.concat(temp_df)
    df = reduce_predicts_same_slice(temp_dataframe)
    # df = temp_df
    return df


def draw_overlay_dicom(pixels,  coord_x, coord_y, coord_z, i, pixel_spacing, dicom_size, png_size, invert_order, png_path):
    z = int(coord_z * png_size[0])
    y = int(coord_y * png_size[1])
    x = int(coord_x * png_size[2])
    dicom_z = int(z / pixel_spacing[2])
    dicom_y = int(y / pixel_spacing[1])
    dicom_x = int(x / pixel_spacing[0])
    x1 = dicom_x - BOX_size
    y1 = dicom_y - BOX_size
    x2 = dicom_x + BOX_size
    y2 = dicom_y + BOX_size
    print("invert_order:", invert_order)
    if invert_order:
        new_z = dicom_z
        org_img = pixels[new_z]
        print("dicom_coord_x_y_z: ", dicom_x, dicom_y, new_z)
    else:
        new_z = dicom_size[0] - dicom_z
        org_img = pixels[new_z]
        # print("dicom_coord_x_y_z: ", dicom_size[2] - dicom_x, dicom_y, new_z)  #for papaya reverse left-right
        print("dicom_coord_x_y_z: ", dicom_x, dicom_y, new_z)

    org_img = normalize_hu_values(org_img)
    cv2.rectangle(org_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    # suffix = i + "_" + str(dicom_size[0] - dicom_z)
    suffix = i + "_" + str(new_z)
    cv2.imwrite(png_path + "/" + "overlay_dicom" + suffix + ".png", org_img * 255)


def get_papaya_coords(coord_x, coord_y, coord_z, nodule_chance, pixel_spacing, dicom_size, png_size, invert_order, ggn_class):
    z = int(coord_z * png_size[0])
    y = int(coord_y * png_size[1])
    x = int(coord_x * png_size[2])
    dicom_z = int(z / pixel_spacing[2])
    dicom_y = int(y / pixel_spacing[1])
    dicom_x = int(x / pixel_spacing[0])

    # print("invert_order:", invert_order)
    if invert_order:
        new_z = dicom_size[0] - dicom_z
        new_x = dicom_size[2] - dicom_x
        new_y = dicom_y
        print("invert_order: ", invert_order, "new_coord_z_y_x: ", new_z, new_y, new_x, " dicom_coord_z_y_x: ", dicom_z, dicom_y, dicom_x)
    else:
        new_z = dicom_size[0] - dicom_z
        new_x = dicom_size[2] - dicom_x
        new_y = dicom_y
        # print("dicom_coord_x_y_z: ", dicom_size[2] - dicom_x, dicom_y, new_z)  #for papaya reverse left-right
        print("invert_order: ", invert_order, "new_coord_z_y_x: ", new_z, new_y, new_x," dicom_coord_z_y_x: ", dicom_z, dicom_y, dicom_x)
    x1 = new_x - BOX_size
    y1 = new_y - BOX_size
    x2 = new_x + BOX_size
    y2 = new_y + BOX_size
    z1 = new_z - BOX_depth
    z2 = new_z + BOX_depth
    box = [z1, y1, x1, z2, y2, x2]
    center = [new_z, new_y, new_x, nodule_chance*100, ggn_class[0], ggn_class[1]*100]
    return box, center


def get_papaya_coords_only(coord_x, coord_y, coord_z, nodule_chance, pixel_spacing, dicom_size, png_size, invert_order):
    z = int(coord_z * png_size[0])
    y = int(coord_y * png_size[1])
    x = int(coord_x * png_size[2])
    dicom_z = int(z / pixel_spacing[2])
    dicom_y = int(y / pixel_spacing[1])
    dicom_x = int(x / pixel_spacing[0])

    # print("invert_order:", invert_order)
    if invert_order:
        new_z = dicom_size[0] - dicom_z
        new_x = dicom_size[2] - dicom_x
        new_y = dicom_y
        print("invert_order: ", invert_order, "new_coord_z_y_x: ", new_z, new_y, new_x, " dicom_coord_z_y_x: ", dicom_z, dicom_y, dicom_x)
    else:
        new_z = dicom_size[0] - dicom_z
        new_x = dicom_size[2] - dicom_x
        new_y = dicom_y
        # print("dicom_coord_x_y_z: ", dicom_size[2] - dicom_x, dicom_y, new_z)  #for papaya reverse left-right
        print("invert_order: ", invert_order, "new_coord_z_y_x: ", new_z, new_y, new_x," dicom_coord_z_y_x: ", dicom_z, dicom_y, dicom_x)
    x1 = new_x - BOX_size
    y1 = new_y - BOX_size
    x2 = new_x + BOX_size
    y2 = new_y + BOX_size
    z1 = new_z - BOX_depth
    z2 = new_z + BOX_depth
    box = [z1, y1, x1, z2, y2, x2]
    center = [new_z, new_y, new_x, round(nodule_chance,3)]
    return box, center


def run(dicom_path, png_save_dir):
    CONTINUE_JOB = True
    sw = Stopwatch.start_new()
    pixel_spacing, dicom_size, png_size, invert_order = extract_dicom_images_patient(dicom_path, png_save_dir)
    final_nodules_df = multipule_test(png_save_dir, CONTINUE_JOB)
    # final_nodules_df = pandas.read_csv(settings_jjyang.OVERLAY_PATH + only_patient_id + ".csv")
    slices = load_dicom_slices(dicom_path)
    pixels = get_pixels_hu(slices)
    i = 0
    for index, row in final_nodules_df.iterrows():
        coord_z = row["coord_z"]
        coord_y = row["coord_y"]
        coord_x = row["coord_x"]
        print("index-x-y-z-p-size", i, coord_x, coord_y, coord_z, row["nodule_chance"], row["diameter_mm"])
        draw_overlay_dicom(pixels, coord_x, coord_y, coord_z, str(i), pixel_spacing, dicom_size, png_size, invert_order)
        # draw_overlay(only_patient_id, coord_x, coord_y, coord_z, str(i))
        i += 1
    # reduce_predicts_same_slice(pred_nodules_df)
    print("ALL Complete in : ", sw.get_elapsed_seconds(), " seconds")


def predict_nodule_type(df, png_size, png_dir):
    list = []
    new_dir = png_dir + '/png/'
    images = load_patient_images(new_dir, "*" + CUBE_IMGTYPE_SRC + ".png")
    i = 0
    for index, row in df.iterrows():
        coord_z = row["coord_z"]
        coord_y = row["coord_y"]
        coord_x = row["coord_x"]
        z = int(coord_z * png_size[0])
        y = int(coord_y * png_size[1])
        x = int(coord_x * png_size[2])
        print("index-x-y-z-p-size-png", x, y, z)
        # print('z invert order : ', invert_order)
        # if not invert_order:
        #     coord_z = int((dicom_size[0] - row["z"]) * pixel_spacing[2])
        # else:
        #     coord_z = int(row["z"] * pixel_spacing[2])
        cube_img = get_cube_from_img(images, x, y, z, 32)
        # save_cube_img('./' + png_dir + '/_' + str(i) + '.png', cube_img, 4, 8)
        save_cube_img(png_dir + '/_' + str(x)+'_'+str(y)+'_'+str(z) + '.png', cube_img, 4, 8)
        img3d = prepare_image_for_net3D(cube_img)
        list.append(img3d)
        i += 1

    img_numpy = numpy.vstack(list)
    return img_numpy

def generate_ggn_class(ii_index):
    if ii_index == 0:
        return 'AAH'
    if ii_index == 1:
        return 'AIS'
    if ii_index == 2:
        return 'MIA'
    if ii_index == 3:
        return 'IA'
    if ii_index == 4:
        return 'OH'


def scan(dicom_path, only_patient_id, workspace, file_type="dicom"):
    """
    扫描节点，支持DICOM和MHD文件
    
    :param dicom_path: DICOM目录或MHD文件路径
    :param only_patient_id: 患者ID
    :param workspace: 工作目录
    :param file_type: 文件类型，'dicom'或'mhd'
    :return: 节点包围盒和中心点
    """
    print("workspace", workspace)
    try:
        target_dir = workspace
        p_index = 0
        # 提取图像
        if file_type == "dicom":
            # 从DICOM提取图像
            pixel_spacing, dicom_size, png_size, invert_order = extract_dicom_images_patient(dicom_path,  target_dir)
        else:
            # 对于MHD文件，图像已经在前面的处理步骤中提取
            pixel_spacing = [1.0, 1.0, 1.0]  # 默认值，后续可根据需要从MHD信息中提取
            dicom_size = [0, 0, 0]  # 默认值
            png_size = [0, 0, 0]    # 默认值
            invert_order = False
            
            # 尝试从PNG目录中获取实际尺寸
            png_dir = os.path.join(target_dir, 'png')
            if os.path.exists(png_dir):
                png_files = glob.glob(png_dir + "/*_i.png")
                if png_files:
                    sample_img = cv2.imread(png_files[0], cv2.IMREAD_GRAYSCALE)
                    png_size = [len(png_files), sample_img.shape[0], sample_img.shape[1]]
        
        # 针对特定模型进行预测
        model_path = os.path.join(current_dir, '../service/workdir/jjnode_classify_resnet_best.h5')
        df_pos = predict_cubes(target_dir, model_path, True, only_patient_id, False, 1, False)
        # 添加分类处理
        df_node = predict_nodule_type(df_pos, png_size, target_dir)
        df_pos = df_node

        # 获取框和中心点信息
        boxes = []
        record_list = []
        records_png_paths = []
        for index, row in df_pos.iterrows():
            record = []
            coord_x = row["coord_x"]
            coord_y = row["coord_y"]
            coord_z = row["coord_z"]
            nodule_chance = float(row["nodule_chance"])
            box_label = row["box_label"]

            # 绘制可视化
            draw_overlay(target_dir, coord_x, coord_y, coord_z, str(p_index))
            
            try:
                # 获取节点坐标
                box = get_papaya_coords(
                    coord_x, coord_y, coord_z, nodule_chance, pixel_spacing, dicom_size, png_size, invert_order, box_label)
                boxes.append(box)

                # 收集节点信息
                record.append(str(p_index))
                record.append(coord_x)
                record.append(coord_y)
                record.append(coord_z)
                record.append(box_label)
                record.append(nodule_chance)
                record_list.append(record)
                
                # 生成立方体图像
                png_f = make_test_cube(record)
                records_png_paths.append(png_f)
                
            except Exception as e:
                print(f"处理节点时出错: {e}")
                
            p_index += 1
            
        return boxes, record_list
        
    except Exception as e:
        print(f"扫描过程中出错: {e}")
        return [], []


def scan_only(dicom_path, only_patient_id, workspace, file_type="dicom"):
    """
    只扫描节点，不分类，支持DICOM和MHD文件
    
    :param dicom_path: DICOM目录或MHD文件路径
    :param only_patient_id: 患者ID
    :param workspace: 工作目录
    :param file_type: 文件类型，'dicom'或'mhd'
    :return: 节点包围盒和中心点
    """
    print("workspace", workspace)
    try:
        target_dir = workspace
        model_path = os.path.join(current_dir, '../service/workdir/jjnode_loc_resnet_best.h5')
        p_index = 0
        
        # 提取图像
        if file_type == "dicom":
            # 从DICOM提取图像
            pixel_spacing, dicom_size, png_size, invert_order = extract_dicom_images_patient(dicom_path, target_dir)
        else:
            # 对于MHD文件，图像已经在前面的处理步骤中提取
            pixel_spacing = [1.0, 1.0, 1.0]  # 默认值，后续可根据需要从MHD信息中提取
            dicom_size = [0, 0, 0]  # 默认值
            png_size = [0, 0, 0]    # 默认值
            invert_order = False
            
            # 尝试从PNG目录中获取实际尺寸
            png_dir = os.path.join(target_dir, 'png')
            if os.path.exists(png_dir):
                png_files = glob.glob(png_dir + "/*_i.png")
                if png_files:
                    sample_img = cv2.imread(png_files[0], cv2.IMREAD_GRAYSCALE)
                    png_size = [len(png_files), sample_img.shape[0], sample_img.shape[1]]
        
        # 预测节点
        df_pos = predict_cubes(target_dir, model_path, True, only_patient_id, False, 1, False)
        
        # 过滤预测结果
        df_pos = filter_patient_nodules_predictions(df_pos, only_patient_id, BOX_size, target_dir)
        df_pos = reduce_predicts_same_slice(df_pos)
        
        # 获取框和中心点信息
        boxes = []
        centers = []
        for index, row in df_pos.iterrows():
            coord_x = row["coord_x"]
            coord_y = row["coord_y"]
            coord_z = row["coord_z"]
            nodule_chance = float(row["nodule_chance"])
            
            # 绘制可视化
            draw_overlay(target_dir, coord_x, coord_y, coord_z, str(p_index))
            
            try:
                # 获取节点坐标
                box = get_papaya_coords_only(
                    coord_x, coord_y, coord_z, nodule_chance, pixel_spacing, dicom_size, png_size, invert_order)
                boxes.append(box)
                
                # 节点中心点（标记百分比）
                z_perc = coord_z
                y_perc = coord_y
                x_perc = coord_x
                nodule_chance_perc = nodule_chance
                diamm = NODULE_DIAMM
                center = [int(z_perc * dicom_size[0]), int(y_perc * dicom_size[1]), int(x_perc * dicom_size[2]), diamm, nodule_chance_perc * 100]
                centers.append(center)
                
            except Exception as e:
                print(f"处理节点时出错: {e}")
                
            p_index += 1
            
        return boxes, centers
        
    except Exception as e:
        print(f"扫描过程中出错: {e}")
        return [], []


# horos_5358
if __name__ == "__main__":
    workspace = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                             './static/workspace/KHAXVYV5')
    scan('E:/renji_hospital_dicom/AIS/ChenZhou/IE2UNXKC/KHAXVYV5',
         'KHAXVYV5', workspace)
    # df = pandas.read_csv("./test.csv")
    # predict_nodule_type(df,[354, 305, 305],'./static/workspace/0708c00f6117ed977bbe1b462b56848c')

