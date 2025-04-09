import numpy,pandas
import os
from util import progress_watch
from detector import extract_dicom_images_patient,get_papaya_coords,prepare_image_for_net3D,predict_nodule_type
from keras.models import load_model, model_from_json
from keras.optimizers import SGD

from util.image_util import load_patient_images, rescale_patient_images
from util.ml.metrics import get_3d_pixel_l2_distance
from util.progress_watch import Stopwatch

PREDICT_STEP = 12
CUBE_SIZE = 32
P_TH = 0.85

def scan(dicom_path, only_patient_id, workspace):
    target_dir = workspace
    boxes = []
    centers = []
    CONTINUE_JOB = True
    sw = Stopwatch.start_new()
    pixel_spacing, dicom_size, png_size, invert_order = extract_dicom_images_patient(dicom_path, target_dir)
    print("png_size: ", png_size)
    final_nodules_df = multipule_test(workspace, only_patient_id, CONTINUE_JOB)
    final_nodules_df_sort = final_nodules_df.sort_values(['nodule_chance'], ascending=False)
    print("predict from maligancy...")
    print("*"*20)
    print(final_nodules_df_sort)
    print("*" * 20)
    if len(final_nodules_df_sort) == 0:
        return boxes, centers

    ggn_npy = predict_nodule_type(final_nodules_df_sort, png_size, workspace)
    json_file = open('./models/c3d_malignancy_regreession.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    model_weight = './models/c3d_malignancy_regreession_04_0.8719.hd5'
    loaded_model.load_weights(model_weight)
    print("Loaded model_0 from disk")
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model_result = loaded_model.predict(ggn_npy, batch_size=20, verbose=1)
    ggn_class_list = []
    for ii in model_result:
        print(ii)
        ii_index = numpy.argmax(ii)
        print("result from malignancy..")
        print(generate_ggn_class(ii_index), round(ii[ii_index], 3))
        # smaller than 0.5 means not malignancy
        if round(ii[ii_index], 3)> 0.5:
            ggn_class_list.append([generate_ggn_class(ii_index), round(ii[ii_index], 3)])

    i = 0
    for index, row in final_nodules_df_sort.iterrows():
        print(ggn_class_list[i])
        coord_z = row["coord_z"]
        coord_y = row["coord_y"]
        coord_x = row["coord_x"]
        print("index-x-y-z-p", coord_x, coord_y, coord_z, row["nodule_chance"])
        box, center = get_papaya_coords(coord_x, coord_y, coord_z, row["nodule_chance"], pixel_spacing, dicom_size,
                                        png_size, invert_order, ggn_class_list[i])

        boxes.append(box)
        centers.append(center)
        # draw_overlay(target_dir, coord_x, coord_y, coord_z, str(i))
        # draw_overlay_dicom(pixels, only_patient_id, coord_x, coord_y, coord_z, str(i), pixel_spacing, dicom_size,
        #                    png_size, invert_order, target_dir)
        i += 1

    print("ALL Complete in : ", sw.get_elapsed_seconds(), " seconds")
    return boxes, centers

def generate_ggn_class(ii_index):
    if ii_index == 0:
        return 'non_malignancy'
    if ii_index == 1:
        return 'malignancy'


def multipule_test(workspace, only_patient_id, CONTINUE_JOB):
    temp_df = []
    for model_version in ["model_loc.hd5", "model_loc_val_0.96.hd5"]:
        print("gpu begin:")
        pred_nodules_df = locate_malignancy(workspace, "models/" + model_version, CONTINUE_JOB, only_patient_id=only_patient_id,
                                        magnification=1, flip=False, train_data=True, holdout_no=None,
                                        ext_name="luna16_fs")
        pred_nodules_df = pred_nodules_df[pred_nodules_df["nodule_chance"] > P_TH]
        temp_df.append(pred_nodules_df)
    temp_dataframe = pandas.concat(temp_df)
    df = reduce_predicts_same_slice(temp_dataframe)
    # df = temp_df
    return df

def locate_malignancy(png_path, model_weight,CONTINUE_JOB, only_patient_id,
                                        magnification=1, flip=False, train_data=True, holdout_no=None,
                                        ext_name="luna16_fs"):
    patient_id = only_patient_id
    all_predictions_csv = []
    sw = helpers.Stopwatch.start_new()
    #json_file = open('/home/xiatao/workspace/renji_dicom/model_json/c3d_5_label_classify.json', 'r')
    json_file = open('../service/workdir/model_loc.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(model_weight)
    patient_img = load_patient_images(png_path + "/png/", "*_i.png", [])
    if magnification != 1:
        patient_img = rescale_patient_images(patient_img, (1, 1, 1), magnification)

    patient_mask = load_patient_images(png_path + "/png/", "*_m.png", [])
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
                            # print("before malignancy_chance")
                            # print(p[0])
                            malignancy_chance = p[0][i][0]
                            predict_volume[p_z, p_y, p_x] = malignancy_chance
                            if malignancy_chance > P_TH:
                                p_z = p_z * step + CROP_SIZE / 2
                                p_y = p_y * step + CROP_SIZE / 2
                                p_x = p_x * step + CROP_SIZE / 2

                                p_z_perc = round(p_z / patient_img.shape[0], 4)
                                p_y_perc = round(p_y / patient_img.shape[1], 4)
                                p_x_perc = round(p_x / patient_img.shape[2], 4)

                                nodule_chance = round(malignancy_chance, 4)
                                patient_predictions_csv_line = [annotation_index, p_x_perc, p_y_perc, p_z_perc,
                                                                 nodule_chance]
                                patient_predictions_csv.append(patient_predictions_csv_line)
                                all_predictions_csv.append([patient_id] + patient_predictions_csv_line)
                                annotation_index += 1

                        batch_list = []
                        batch_list_coords = []
                done_count += 1
                if done_count % 10000 == 0:
                    print("Scan: ", done_count, " skipped:", skipped_count)

    df = pandas.DataFrame(patient_predictions_csv,
                          columns=["anno_index", "coord_x", "coord_y", "coord_z",  "nodule_chance"])

    filter_patient_nodules_predictions(df, patient_id, CROP_SIZE * magnification, png_path)

    print(predict_volume.mean())
    print("GPU costs : ", sw.get_elapsed_seconds(), " seconds")
    return df

def filter_patient_nodules_predictions(df_nodule_predictions: pandas.DataFrame, patient_id, view_size, png_path):
    patient_mask = load_patient_images(png_path+"/png/", "*_m.png")
    delete_indices = []
    for index, row in df_nodule_predictions.iterrows():
        z_perc = row["coord_z"]
        y_perc = row["coord_y"]
        center_x = int(round(row["coord_x"] * patient_mask.shape[2]))
        center_y = int(round(y_perc * patient_mask.shape[1]))
        center_z = int(round(z_perc * patient_mask.shape[0]))

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
            #delete_indices.append(df_nodule_predictions.loc[index,])
            delete_indices.append(index)
        else:
            if center_z < 30:
                print("Z < 30: ", patient_id, " center z:", center_z, " y_perc: ", y_perc)
                #delete_indices.append(df_nodule_predictions.loc[index])
                delete_indices.append(index)

            if (z_perc > 0.75 or z_perc < 0.25) and y_perc > 0.85:
                print("SUSPICIOUS FALSEPOSITIVE: ", patient_id, " center z:", center_z, " y_perc: ", y_perc)
                #delete_indices.append(df_nodule_predictions.loc[index])
                delete_indices.append(index)

            if center_z < 50 and y_perc < 0.30:
                print("SUSPICIOUS FALSEPOSITIVE OUT OF RANGE: ", patient_id, " center z:", center_z, " y_perc: ",
                      y_perc)
                #delete_indices.append(df_nodule_predictions.loc[index])
                delete_indices.append(index)
    print("slice to drop:\t",delete_indices)
    df_nodule_predictions.drop(df_nodule_predictions.index[delete_indices], inplace=True)
    return df_nodule_predictions

def reduce_predicts_same_slice(pred_nodules_df):
    rows_filter = []
    pred_nodules_df_local = pred_nodules_df.sort_values(["coord_z"], ascending=False)
    if len(pred_nodules_df_local) <= 1:
        return pred_nodules_df_local
    compare_row = pred_nodules_df_local.iloc[0]
    for row_index, row in pred_nodules_df_local[1:].iterrows():
        if compare_row["coord_z"] == row["coord_z"]:
            dist = get_3d_pixel_l2_distance(compare_row, row)
            if dist > 0.2:
                rows_filter.append(row)
        else:
            rows_filter.append(compare_row)
            compare_row = row
    if len(rows_filter) == 0:
        rows_filter.append(compare_row)
    last_row = rows_filter[len(rows_filter)-1]
    if last_row["coord_z"] != compare_row["coord_z"]:
        rows_filter.append(compare_row)
    columns = ["anno_index", "coord_x", "coord_y", "coord_z", "nodule_chance"]
    res_df = pandas.DataFrame(rows_filter, columns=columns)
    return res_df