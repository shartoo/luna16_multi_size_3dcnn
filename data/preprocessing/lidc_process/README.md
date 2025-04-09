# LIDC-IDRI 数据预处理

本目录包含处理 [LIDC-IDRI (Lung Image Database Consortium and Image Database Resource Initiative)](https://www.cancerimagingarchive.net/collection/lidc-idri/) 数据集的预处理脚本。LIDC-IDRI 是一个公开的肺部CT扫描数据集，包含了1000多个患者的胸部CT扫描和相应的医生标注。

## LIDC-IDRI 数据集详情

LIDC-IDRI 数据集由7个学术中心和8个医学影像公司合作创建，共包含1018个案例。每个案例包括临床胸部CT扫描图像和由四位有经验的胸部放射科医生进行的标注结果。标注过程分为两个阶段：

1. **盲审阶段**：每位放射科医生独立审查每个CT扫描，并标记属于三个类别之一的病变：
   - 直径≥3mm的结节
   - 直径<3mm的结节
   - 直径≥3mm的非结节

2. **非盲审阶段**：每位放射科医生独立审查自己的标记以及其他三位放射科医生的匿名标记，以形成最终意见。

### 数据组成

- **CT扫描**：1018个病例的胸部CT扫描DICOM文件
- **标注XML文件**：包含结节位置、大小和特征的XML格式标注
- **结节诊断信息**：包含结节的恶性度评分和其他特征

### 标注特征说明

每个标记的结节包含以下特征评分（1-5分）：

| 特征名称 | 评分范围 | 含义 |
| ------- | ------- | ---- |
| 恶性度(malignancy) | 1-5 | 1=高度良性，5=高度恶性 |
| 球形度(sphericity) | 1-5 | 1=线性，5=完全球形 |
| 边缘特征(margin) | 1-5 | 1=明显，5=模糊 |
| 毛刺(spiculation) | 1-5 | 1=无毛刺，5=明显毛刺 |
| 纹理(texture) | 1-5 | 1=非实性，5=实性 |
| 钙化(calcification) | 1-6 | 不同类型的钙化 |
| 内部结构(internal structure) | 1-4 | 不同类型的内部结构 |
| 分叶性(lobulation) | 1-5 | 1=无分叶，5=明显分叶 |
| 细微性(subtlety) | 1-5 | 1=明显，5=细微 |

## 数据示例

### XML标注示例

```xml
<LidcReadMessage>
  <ResponseHeader>
    <SeriesInstanceUid>1.3.6.1.4.1.14519.5.2.1.6279.6001.123456789</SeriesInstanceUid>
  </ResponseHeader>
  <readingSession>
    <servicingRadiologistID>Reader1</servicingRadiologistID>
    <unblindedReadNodule>
      <noduleID>Nodule001</noduleID>
      <characteristics>
        <malignancy>4</malignancy>
        <sphericity>5</sphericity>
        <margin>4</margin>
        <spiculation>3</spiculation>
        <texture>5</texture>
        <calcification>1</calcification>
        <internalStructure>1</internalStructure>
        <lobulation>2</lobulation>
        <subtlety>3</subtlety>
      </characteristics>
      <roi>
        <imageZposition>-124.0</imageZposition>
        <edgeMap>
          <xCoord>256</xCoord>
          <yCoord>215</yCoord>
        </edgeMap>
        <!-- 更多边缘点... -->
      </roi>
      <!-- 更多ROI... -->
    </unblindedReadNodule>
    <nonNodule>
      <nonNoduleID>NonNodule001</nonNoduleID>
      <imageZposition>-134.0</imageZposition>
      <locus>
        <xCoord>345</xCoord>
        <yCoord>287</yCoord>
      </locus>
    </nonNodule>
  </readingSession>
  <!-- 更多readingSession... -->
</LidcReadMessage>
```

### 处理后的CSV数据示例

**百分比坐标CSV（process_lidc_annotations输出）**：

```
patient_id,anno_index,servicingRadiologistID,coord_x,coord_y,coord_z,diameter,malscore,sphericiy,margin,spiculation,texture,calcification,internal_structure,lobulation,subtlety
1.3.6.1.4.1.14519.5.2.1.6279.6001.123456789,Nodule001,Reader1,0.5242,0.4455,0.3789,0.0521,4,5,4,3,5,1,1,2,3
1.3.6.1.4.1.14519.5.2.1.6279.6001.123456789,Nodule001,Reader2,0.5256,0.4478,0.3802,0.0534,3,4,3,2,5,1,1,3,2
```

**毫米坐标CSV（percent_coordinatecsv_to_mmcsv输出）**：

```
patient_id,anno_index,servicingRadiologistID,coord_x,coord_y,coord_z,mm_x,mm_y,mm_z,diameter,malscore,sphericiy,margin,spiculation,texture,calcification,internal_structure,lobulation,subtlety
1.3.6.1.4.1.14519.5.2.1.6279.6001.123456789,Nodule001,Reader1,0.5242,0.4455,0.3789,126.5,107.8,-124.0,0.0521,4,5,4,3,5,1,1,2,3
1.3.6.1.4.1.14519.5.2.1.6279.6001.123456789,Nodule001,Reader2,0.5256,0.4478,0.3802,127.0,108.5,-124.2,0.0534,3,4,3,2,5,1,1,3,2
```

**带平均坐标和恶性度标签的CSV（最终输出）**：

```
patient_id,anno_index,servicingRadiologistID,coord_x,coord_y,coord_z,mm_x,mm_y,mm_z,avg_x,avg_y,avg_z,diameter,malscore,real_mal,sphericiy,margin,spiculation,texture,calcification,internal_structure,lobulation,subtlety
1.3.6.1.4.1.14519.5.2.1.6279.6001.123456789,Nodule001,Reader1,0.5242,0.4455,0.3789,126.5,107.8,-124.0,126.75,108.15,-124.1,0.0521,4,1,5,4,3,5,1,1,2,3
1.3.6.1.4.1.14519.5.2.1.6279.6001.123456789,Nodule001,Reader2,0.5256,0.4478,0.3802,127.0,108.5,-124.2,126.75,108.15,-124.1,0.0534,3,1,4,3,2,5,1,1,3,2
```

### 结节标注统计

在LIDC-IDRI数据集中：
- 共有1018个病例
- 约2669个被至少一位放射科医生标注的≥3mm结节
- 约928个被所有四位放射科医生标注的≥3mm结节
- 约479个被标记为恶性的结节（平均恶性度评分>3）
- 约591个被标记为良性的结节（平均恶性度评分<3）
- 约858个具有不确定恶性度的结节（平均恶性度评分=3）

## 数据处理流程

数据处理分为以下几个主要步骤：

1. 从原始XML标注文件中提取结节信息
2. 将原始百分比坐标转换为毫米坐标
3. 汇总和整合多位放射科医生的结节标注
4. 计算结节的恶性度标签
5. 生成用于模型训练的数据集

## 脚本说明

### 1. lidc_annotation_process.py

该脚本用于处理LIDC-IDRI数据集中的XML标注文件，提取结节的位置、大小和特征信息。

**主要功能**：

- `read_nodule_annotation_from_xml()`: 读取单个XML标注文件，提取结节信息
- `process_lidc_annotations()`: 处理所有XML标注文件，汇总所有结节信息
- `extract_lidc_every_z_annotations()`: 提取每个Z轴切片上的标注信息
- `merge_nodule_annotation_csv_to_one()`: 将多个结节标注CSV文件合并为一个

**处理的信息包括**：

- 结节的位置坐标（百分比形式）
- 结节直径
- 结节恶性度评分（1-5）
- 其他特征：球形度、边缘特征、毛刺、纹理、钙化、内部结构、分叶性和细微性

### 2. lidc_coordinate_process.py

该脚本处理由 `lidc_annotation_process.py` 生成的结果，主要关注标注坐标的转换和处理。

**主要功能**：

- `percent_coordinatecsv_to_mmcsv()`: 将百分比坐标转换为毫米坐标
- `avg_coordinates()`: 计算相同结节的平均坐标
- `add_final_mals()`: 计算每个结节的最终恶性度标签
- `draw_percent_cube_by_csv()`: 根据百分比坐标绘制立方体区域
- `draw_all_confirmed_cubes()`: 绘制所有确认的结节立方体

## 处理流程

1. **读取原始XML标注**:
   - 解析XML文件提取每个放射科医生的标注
   - 获取结节的位置坐标（以图像百分比表示）
   - 提取结节的特征信息（恶性度评分、纹理等）

2. **坐标转换**:
   - 将百分比坐标转换为毫米坐标
   - 根据DICOM元数据计算实际物理位置

3. **结节汇总**:
   - 同一结节可能被多位放射科医生标注
   - 计算相近结节的平均坐标
   - 合并多位医生对同一结节的标注

4. **恶性度计算**:
   - 每个结节由多位医生评分（1-5分）
   - 根据各评分计算最终恶性度标签
   - 0 = 良性，1 = 恶性，"unknow" = 不确定

5. **数据可视化**:
   - 在原始CT图像上绘制结节立方体
   - 用于验证标注准确性

## 使用示例

```python
# 处理XML标注
process_lidc_annotations("path/to/xml/*.xml", patient_mhd_path_dict, "path/to/save/all_annotations.csv")

# 转换坐标
percent_coordinatecsv_to_mmcsv("all_annotations.csv", mhd_info_csv, "mm_coordinates.csv")

# 计算平均坐标
avg_coordinates("mm_coordinates.csv", threshold=5, "avg_coordinates.csv")

# 添加最终恶性度标签
add_final_mals("avg_coordinates.csv", "final_annotations.csv")
```

## 数据集参考

LIDC-IDRI数据集：[https://www.cancerimagingarchive.net/collection/lidc-idri/](https://www.cancerimagingarchive.net/collection/lidc-idri/) 