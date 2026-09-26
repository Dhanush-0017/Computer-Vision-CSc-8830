# Where the Module 4 images come from

## RGB — Penn-Fudan Pedestrian Database

`rgb/*.png` is one image from the Penn-Fudan Pedestrian Database (170
images of pedestrians, photographed around the University of Pennsylvania
and Fudan University).

> Liming Wang, Jianbo Shi, Gang Song, I-fan Shen. *Object Detection Combining
> Recognition and Segmentation.* ACCV 2007.
> https://www.cis.upenn.edu/~jshi/ped_html/

Copied from the GitHub mirror at github.com/swallan/PennFudanPed.

The box in `prompts.json` is the dataset's own annotation box, grown by
5% per side so the person is inside with a little margin (GrabCut and SAM2
both assume that).

| File | Why this one |
|---|---|
| FudanPed00013 | one pedestrian, full body, plain background |

## Thermal — RoadScene (FLIR ADAS frames)

`thermal/*.png` is one infrared frame from RoadScene, which is 221 aligned
infrared/visible pairs taken from the FLIR ADAS driving dataset (FLIR Tau2
thermal camera, 640×512).

> Han Xu, Jiayi Ma, Zhuliang Le, Junjun Jiang, Xiaojie Guo. *FusionDN: A
> Unified Densely Connected Network for Image Fusion.* AAAI 2020.
> https://github.com/jiayi-ma/RoadScene

The box in `prompts.json` I drew by hand.

| File | Why this one |
|---|---|
| FLIR_07583 | person from behind, full body, road warming towards the camera |

## SAM2 masks

`sam2_masks/{rgb,thermal}/*.png` were made by `../sam2_reference.py` with
SAM2.1 Hiera-Large (the `sam2.1_l.pt` checkpoint via the `ultralytics`
package), prompted with the same box as the classical methods. White is
person.

> Nikhila Ravi et al. *SAM 2: Segment Anything in Images and Videos.* 2024.
> https://ai.meta.com/research/sam2/
