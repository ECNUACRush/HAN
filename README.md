# HYATT-Net is Grand: A Hybrid Attention Network for Performant Anatomical Landmark Detection

**This is the official pytorch implementation repository of our HYATT-Net is Grand: A Hybrid Attention Network for Performant Anatomical Landmark Detection** 

- Link to paper: https://arxiv.org/abs/2412.06499


## Dataset
- We have used the following datasets:
  - **ISBI2015  dataset**: Ching-Wei Wang, Cheng-Ta Huang, Jia-Hong Lee, Chung-Hsing Li,Sheng-Wei Chang, Ming-Jhih Siao, Tat-Ming Lai, Bulat Ibragimov,Tomaž Vrtovec, Olaf Ronneberger, et al. A benchmark for comparison of dental radiography analysis algorithms. Medical image analysis,31:63–76, 2016
  - **ISBI2023  dataset**: Anwaar Khalid, M., Zulfiqar, K., Bashir, U., Shaheen, A., Iqbal, R., Rizwan, Z., Rizwan, G., Moazam Fraz, M.: Cepha29: Automatic cephalometric landmark detection challenge 2023. arXiv e-prints pp. arXiv–2212 (2022)
  - **CephAdoAdu**: Han Wu, Chong Wang, Lanzhuju Mei, Tong Yang, Min Zhu, Ding-gang Shen, and Zhiming Cui. Cephalometric landmark detection
    across ages with prototypical network. In International Conference on Medical Image Computing and Computer-Assisted Intervention,pages 155–165. Springer, 2024.
  - **Hand X-Rays**: Payer, C., ˇStern, D., Bischof, H., Urschler, M.: Integrating spatial configuration into heatmap regression based CNNs for landmark localization. Medical Image Analysis 54, 207–219 (2019)
  - **Pelvic X-Rays**: https://www.kaggle.com/datasets/tommyngx/cgmh-pelvisseg；Chi-Tung Cheng, Yirui Wang, Huan-Wu Chen, Po-Meng Hsiao,Chun-Nan Yeh, Chi-Hsun Hsieh, Shun Miao, Jing Xiao, Chien-Hung Liao, and Le Lu. A scalable physician-level deep learning algorithm detects universal trauma on pelvic radiographs. Nature communications, 12(1):1066, 2021.


## Prerequesites
- Python 3.7
- MMpose 0.23

## Usage of the code
- **Dataset preparation**
  - The dataset structure should be in the following structure:

  ```
  Inputs: .PNG images and JSON file
  └── <dataset name>
      ├── 2D_images
      |   ├── 001.png
      │   ├── 002.png
      │   ├── 003.png
      │   ├── ...
      |
      └── JSON
          ├── train.json
          └── test.json
  ```
  - Example json format
  ```
   {
      "images": [
          {
              "id": 0,
              "file_name": "0.png",
              "height": 512,
              "width": 512
          },
          ...
       ],
       "annotations": [
          {
              "image_id": 0,
              "id": 0,
              "category_id": 1,
              "keypoints": [
                  492.8121081534473,
                  261.62600827462876,
                  2,
                  428.0154934462112,
                  301.24809471563935,
                  2,
                  504.9223114040644,
                  234.45993184950234,
                  2,
                  470.296873380625,
                  182.90429052972533,
                  2,
                  456.97751121306436,
                  208.8105499707776,
                  2,
                  369.95414168150415,
                  239.07609878665616,
                  2,
                  307.83364934785106,
                  229.91052362204155,
                  2,
                  373.5995213621739,
                  353.599939601835,
                  2,
                  499.50552505239256,
                  453.1111418891231,
                  2,
                  493.50543334239256,
                  456.12341418891231,
                  2
              ],
              "num_keypoints": 10,
              "iscrowd": 0
          },
          ...
     ]
  ```
  
  - Output: 2D landmark coordinates
  
- **Train the model**
  - To train our model run **sh train.sh**:
  ```
  # sh train.sh
  CUDA_VISIBLE_DEVICES=gpu_ids PORT=PORT_NUM ./tools/dist_train.sh \
  config_file_path num_gpus
  ```

- **Evaluation**
  - To evaluate the trained model run **sh test.sh**:
  ```
  # sh test.sh
  CUDA_VISIBLE_DEVICES=gpu_id PORT=29504 ./tools/dist_test.sh config_file_path \
      model_weight_path num_gpus \
      # For evaluation of the Head XCAT dataset, use:
      --eval 'MRE_h','MRE_std_h','SDR_2_h','SDR_2.5_h','SDR_3_h','SDR_4_h'
      # For evaluation of ISBI2023 and Hand X-ray dataset, use:
      # --eval 'MRE_i2','MRE_std_i2','SDR_2_i2','SDR_2.5_i2','SDR_3_i2','SDR_4_i2'
  ```

