# g32
python train_SCOPSP.py \
-s log2021/0515/SCOPSP_newnorm_wildceleba_9part_perp-1.0_arc-1.0_delta-0.01 \
-c configs/UPS/SCOPSP_WILDCELEBA_newNorm_128x128_TPS_PERCEPNEW_deltaloss_hm32_newtc.py \
--options metriclist_train.-1.weight=0.01 metriclist_train.-2.weight=1.0 \
-g 1

python train_SCOPSP.py \
-s log2021/0423/SCOPSP_newnorm_wildceleba_deltaloss_5part_arc-1.0 \
-c configs/UPS/SCOPSP_WILDCELEBA_newNorm_128x128_TPS_PERCEPNEW_deltaloss_hm32_newtc.py \
--options metriclist_train.-1.weight=0 metriclist_train.-2.weight=1.0 metriclist_train.0.weight=0 \
-g 7

python train_SCOPSP.py \
-s log2021/0423/SCOPSP_newnorm_wildceleba_deltaloss_5part_arc-1.0_delta-0.0001 \
-c configs/UPS/SCOPSP_WILDCELEBA_newNorm_128x128_TPS_PERCEPNEW_deltaloss_hm32_newtc.py \
--options metriclist_train.-1.weight=0.0001 metriclist_train.-2.weight=1.0 metriclist_train.0.weight=0 \
-g 6


# g28
python train_SCOPSP.py \
-s log2021/0410/SCOPSP_wildceleba_5part_perp-1.0_fg-0.1_bg-0.1 \
-c configs/UPS/SCOPSP_WILDCELEBA_Norm_128x128_TPS_PERCEPNEW_concenloss_hm32_newtc.py \
--options metriclist_train.-1.weight=0.1 metriclist_train.-2.weight=0.1 \
-g 2

python train_SCOPSP.py \
-s log2021/0410/SCOPSP_wildceleba_5part_perp-1.0_fg-0.5_bg-0.5 \
-c configs/UPS/SCOPSP_WILDCELEBA_Norm_128x128_TPS_PERCEPNEW_concenloss_hm32_newtc.py \
--options metriclist_train.-1.weight=0.5 metriclist_train.-2.weight=0.5 \
-g 3



# g28
python train_SCOPSP.py \
-s log2021/0410/SCOPSP_wildceleba_deltaloss_5part_perp-1.0_delta2-0.01 \
-c configs/UPS/SCOPSP_WILDCELEBA_Norm_128x128_TPS_PERCEPNEW_deltaloss_hm32_newtc.py \
--options metriclist_train.-1.weight=0.01 \
-g 2

python train_SCOPSP.py \
-s log2021/0410/SCOPSP_wildceleba_deltaloss_5part_perp-1.0_delta2-0.001 \
-c configs/UPS/SCOPSP_WILDCELEBA_Norm_128x128_TPS_PERCEPNEW_deltaloss_hm32_newtc.py \
--options metriclist_train.-1.weight=0.001 \
-g 3




# python train_SCOPSP.py \
# -s log2021/0410/SCOPSP_wildceleba_nonorm_deltaloss_perp-1.0_del-1.0 \
# -c configs/UPS/SCOPSP_WILDCELEBA_Norm_128x128_TPS_PERCEPNEW_deltaloss_hm32_newtc.py \
# --options MODEL.paras.use_norm=0 \
# -g 1